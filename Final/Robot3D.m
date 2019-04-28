 classdef Robot3D 
    properties (SetAccess = 'immutable')
        BASE_HEIGHT = 6.3;
        LINK_1 = 3.1;
        LINK_2 = 42.7;
        LINK_3 = 43;
        LINK_4 = 10;
    end
    
    methods
        function result = translation_x(robot, x)
            result = [1,0,0,x;
                      0,1,0,0;
                      0,0,1,0;
                      0,0,0,1];
        end
        
        function result = translation_y(robot, y)
            result = [1,0,0,0;
                      0,1,0,y;
                      0,0,1,0;
                      0,0,0,1];
        end
        
        function result = translation_z(robot, z)
            result = [1,0,0,0;
                      0,1,0,0;
                      0,0,1,z;
                      0,0,0,1];
        end
        
        function result = rotation_x(robot, theta)
            result = [1,0,0,0;
                      0,cos(theta),-sin(theta),0;
                      0,sin(theta),cos(theta),0;
                      0,0,0,1];
        end
        
        function result = rotation_y(robot, theta)
            result = [cos(theta),0,sin(theta),0;
                      0,1,0,0;
                      -sin(theta),0,cos(theta),0;
                      0,0,0,1];
        end
        
        function result = rotation_z(robot, theta)
            result = [cos(theta),-sin(theta),0,0;
                      sin(theta),cos(theta),0,0;
                      0,0,1,0;
                      0,0,0,1];
        end

        function frames = forward_kinematics(robot, thetas)
            frames = zeros(4,4,6);
            
            theta = thetas(1);
            frames(:,:,1) = robot.rotation_z(theta);
            
            theta = thetas(2);
            frames(:,:,2) = frames(:,:,1) * robot.translation_z(robot.LINK_1+robot.BASE_HEIGHT) * robot.rotation_y(-theta) ;
            
            theta = thetas(3);
            frames(:,:,3) = frames(:,:,2) * robot.translation_x(robot.LINK_2) * robot.rotation_y(theta);
            
            theta = thetas(4);
            frames(:,:,4) = frames(:,:,3) * robot.translation_x(robot.LINK_3) * robot.rotation_y(theta);
            
            theta = thetas(5);
            frames(:,:,5) = frames(:,:,4) * robot.rotation_x(theta);
            
            frames(:,:,6) = frames(:,:,5) * robot.translation_x(robot.LINK_4);
        end
        
        function goal_angles = numerical_IK(robot, goal_position, initial_theta)
            function err = my_error_function(theta)   
              actual_pos = robot.end_effector(theta);
              actual_position = actual_pos(1:3);
              goal_position = goal_position(1:3);
              err = (goal_position - actual_position).^2;
              err = sum(err);
            end
           
            % Actually run the optimization to generate the angles to get us (close) to
            % the goal.
            % Set joint limit for better performance
            lb = [-pi/2, 0, 0, 0, -pi/4];
            ub = [pi/2, pi, pi, pi/2, pi/4];
            
            %Final goal_angles
            goal_angles = fmincon( @ my_error_function,initial_theta,[],[],[],[],lb,ub);
            goal_angles = wrapToPi(goal_angles);
        end


        %% Shorthand for returning the forward kinematics.
        function fk = fk(robot, thetas)
            fk = robot.forward_kinematics(thetas);
        end
       
        % Returns [x; y; theta] for the end effector given a set of joint
        % angles. 
        function ee = end_effector(robot, thetas)
            % Find the transform to the end-effector frame.
            frames = robot.fk(thetas);
            H_0_ee = frames(:,:,end);
            
            % Extract the components of the end_effector position and
            % orientation.
            x = H_0_ee(1,4);
            y = H_0_ee(2,4);
            z = H_0_ee(3,4);
            roll = atan2(H_0_ee(2, 1), H_0_ee(1, 1));
            pitch = -asin(H_0_ee(3, 1));
            yaw = atan2(H_0_ee(3, 2), H_0_ee(3, 3));
            % Pack them up nicely.
            ee = [x; y; z; roll; pitch; yaw];
        end
       
        %% Shorthand for returning the end effector position and orientation. 
        function ee = ee(robot, thetas)
            ee = robot.end_effector(thetas);
        end
    end
end