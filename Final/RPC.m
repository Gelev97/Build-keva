% Open Socket
ip = '0.0.0.0';
port = 2002;
fprintf('Socket openning @%s:%d ... ',ip,port);
t = tcpip(ip, port, 'NetworkRole','server');
fopen(t);
fprintf('Opened.\n');

robot = Robot3D();

while true
    % Read from socket (wait here)
    while t.BytesAvailable == 0, WAIT=true; end
    data = fread(t, t.BytesAvailable);
    string = char(data)';
    newStr = split(string,',');
    goal_position = [str2double(newStr(1));str2double(newStr(2));
                     str2double(newStr(3))]
    initial_theta = [str2double(newStr(4)),str2double(newStr(5)),str2double(newStr(6))];
    theta = robot.numerical_IK(goal_position,initial_theta);
    calculated_location = robot.ee(theta)
    % Send to socket
    tx_data = sprintf('%f,%f,%f',theta(1),theta(2),theta(3));
    fwrite(t, tx_data);
    
    % terminate 
    pause(1e-0);
end

% Close
fclose(t);
fprintf('Closed.\n');


