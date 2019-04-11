#define CMD_GET_BATTERY_VOLTAGE 0x0F
// 0x55 0x55 0x02 CMD_GET_BATTERY_VOLTAGE
#define CMD_MULT_SERVO_UNLOAD 20
// 0x55 0x55 3+N CMD_MULT_SERVO_UNLOAD N 1 2 ... 5 6
#define CMD_SERVO_MOVE 3
// 0x55 0x55 5+N*3 CMD_SERVO_MOVE N t0 t1 1 p0 p1 2 p0 p1 3 p0 p1 ... 6 p0 p1
#define CMD_MULT_SERVO_POS_READ 21
// 0x55 0x55 3+N CMD_MULT_SERVO_POS_READ N 1 2 ... 5 6
#define CMD_MULT_SERVO_POS_WRITE 22

#define DATA_BATTERY_VOLTAGE CMD_GET_BATTERY_VOLTAGE
// 0x55 0x55 0x04 CMD_GET_BATTERY_VOLTAGE V0 V1
#define DATA_MULT_SERVO_POS_READ CMD_MULT_SERVO_POS_READ
// 0x55 0x55 3+N*3 CMD_MULT_SERVO_POS_READ N 1 p0 p1 2 ... 6 p0 p1
