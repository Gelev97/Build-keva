/**********************************************************************/
// SPDX-License-Identifier: GPL-2.0
/*
 * Hidraw Userspace Example
 *
 * Copyright (c) 2010 Alan Ott <alan@signal11.us>
 * Copyright (c) 2010 Signal 11 Software
 *
 * The code may be used by anyone for any purpose,
 * and can serve as a starting point for developing
 * applications using hidraw.
 */

/**********************************************************************/

/* Linux */
#include <linux/types.h>
#include <linux/input.h>
#include <linux/hidraw.h>

/* Unix */
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>

/* C */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h> 

#include "xarm.h"

/**********************************************************************/

char *device = "/dev/hidraw2";

// has a request been sent and not answered?
int get_battery_voltage_pending = 0;
// latest value
int battery_voltage = 0;

int read_angles_pending = 0;
int angles[ 10 ]; // right now joints are tip = 3, 4, 5, 6 = base

int angles_desired[ 10 ]; // right now joints are tip = 3, 4, 5, 6 = base

/**********************************************************************/
/**********************************************************************/

const char *
bus_str(int bus)
{
	switch (bus) {
	case BUS_USB:
		return "USB";
		break;
	case BUS_HIL:
		return "HIL";
		break;
	case BUS_BLUETOOTH:
		return "Bluetooth";
		break;
	case BUS_VIRTUAL:
		return "Virtual";
		break;
	default:
		return "Other";
		break;
	}
}

/**********************************************************************/
/**********************************************************************/
/**********************************************************************/

void init_hidraw( char *device, int *pfd )
{
  int fd;
  int i, res, desc_size = 0;
  char buf[256];
  struct hidraw_report_descriptor rpt_desc;
  struct hidraw_devinfo info;

  /* Open the Device with non-blocking reads. In real life,
     don't use a hard coded path; use libudev instead. */
  *pfd = fd = open( device, O_RDWR | O_NONBLOCK );

  if (fd < 0)
    {
      perror("Unable to open device");
      exit( -1 );
    }

  memset(&rpt_desc, 0x0, sizeof(rpt_desc));
  memset(&info, 0x0, sizeof(info));
  memset(buf, 0x0, sizeof(buf));

  /* Get Report Descriptor Size */
  res = ioctl(fd, HIDIOCGRDESCSIZE, &desc_size);
  if (res < 0)
    perror("HIDIOCGRDESCSIZE");
  /*
  else
    printf("Report Descriptor Size: %d\n", desc_size);
  */

  /* Get Report Descriptor */
  rpt_desc.size = desc_size;
  res = ioctl(fd, HIDIOCGRDESC, &rpt_desc);
  if (res < 0)
    {
      perror("HIDIOCGRDESC");
    }
  /*
  else
    {
      printf("Report Descriptor:\n");
      for (i = 0; i < rpt_desc.size; i++)
	printf("%hhx ", rpt_desc.value[i]);
      puts("\n");
    }
  */

  /* Get Raw Name */
  res = ioctl(fd, HIDIOCGRAWNAME(256), buf);
  if (res < 0)
    perror("HIDIOCGRAWNAME");
  else
    printf("Raw Name: %s\n", buf);

  if ( strstr( buf, "LOBOT" ) == NULL )
    {
      printf( "\nIf the above line does not say\n" );
      printf( "Raw Name: MyUSB_HID LOBOT\n" );
      printf( "You need to change the line 'char *device = \"%s\";'\n", device );
      printf( "Look for the following line in the output of\n" );
      printf( "dmesg | tail\n" );
      printf( "   output looks like:\n" );
      printf( "[ timestamp ] hid-generic ... hiddev96,hidraw2: ... [MyUSB_HID LOBOT] on ...\n" );
      printf( " The hidrawX tells you which device is actually in use\n\n" );
    }

  /* Get Physical Location */
  res = ioctl(fd, HIDIOCGRAWPHYS(256), buf);
  if (res < 0)
    perror("HIDIOCGRAWPHYS");
  /*
  else
    printf("Raw Phys: %s\n", buf);
  */

  /* Get Raw Info */
  res = ioctl(fd, HIDIOCGRAWINFO, &info);
  if (res < 0)
    {
      perror("HIDIOCGRAWINFO");
    }
  /*
  else
    {
      printf("Raw Info:\n");
      printf("\tbustype: %d (%s)\n",
	     info.bustype, bus_str(info.bustype));
      printf("\tvendor: 0x%04hx\n", info.vendor);
      printf("\tproduct: 0x%04hx\n", info.product);
    }
  */
}

/**********************************************************************/
/**********************************************************************/

// get battery voltage
#define N_BATTERY_VOLTAGE 5
char BATTERY_VOLTAGE[ N_BATTERY_VOLTAGE ] = {
  0,
  0x55,
  0x55,
  0x02,
  CMD_GET_BATTERY_VOLTAGE
};

/**********************************************************************/

void get_battery_voltage( int fd )
{
  int wlen;

  wlen = write( fd, BATTERY_VOLTAGE, N_BATTERY_VOLTAGE );
  if ( wlen != N_BATTERY_VOLTAGE )
    {
      printf( "Error from write: %d != %d; errno = %d\n", wlen,
	      N_BATTERY_VOLTAGE, errno );
      exit( -1 );
    }
  get_battery_voltage_pending = 1;
  // printf( "wrote: %d\n", wlen );
}

/**********************************************************************/

int parse_battery_voltage( unsigned char buf[] )
{
  if ( buf[ 2 ] != 0x04 )
    {
      printf( "bad battery voltage response: 0x%x\n", buf[ 2 ] );
      return 0;
    }
  battery_voltage = buf[ 3 ] + (buf[ 4 ] << 8);
  printf( "battery voltage: %d (%d)\n",
	  battery_voltage, get_battery_voltage_pending );
  get_battery_voltage_pending = 0;
  return 1;
}

/**********************************************************************/
/**********************************************************************/

// get angles
#define N_READ_ANGLES 10
char READ_ANGLES[ N_READ_ANGLES ] = {
  0,
  0x55,
  0x55,
  0x06,
  CMD_MULT_SERVO_POS_READ,
  0x03,
  0x04,
  0x05,
  0x06
};

/**********************************************************************/

void get_angles( int fd )
{
  int wlen;

  wlen = write( fd, READ_ANGLES, N_READ_ANGLES );
  if ( wlen != N_READ_ANGLES )
    {
      printf( "Error from write: %d != %d; errno = %d\n", wlen,
	      N_READ_ANGLES, errno );
      exit( -1 );
    }
  read_angles_pending = 1;
  // printf( "wrote: %d\n", wlen );
}

/**********************************************************************/

int parse_read_angles( unsigned char buf[] )
{
  if ( buf[ 2 ] != 0x0C )
    {
      printf( "bad read_angles response 1: 0x%x\n", buf[ 2 ] );
      return 0;
    }
  if ( buf[ 4 ] != 0x03 )
    {
      printf( "bad read_angles response 2: 0x%x\n", buf[ 4 ] );
      return 0;
    }
  if ( buf[ 5 ] != 0x04 )
    {
      printf( "bad read_angles response 4: 0x%x\n", buf[ 5 ] );
      return 0;
    }
  angles[ 4 ] = buf[ 6 ] + (buf[ 7 ] << 8);
  if ( buf[ 8 ] != 0x05 )
    {
      printf( "bad read_angles response 5: 0x%x\n", buf[ 8 ] );
      return 0;
    }
  angles[ 5 ] = buf[ 9 ] + (buf[ 10 ] << 8);
  if ( buf[ 11 ] != 0x06 )
    {
      printf( "bad read_angles response 6: 0x%x\n", buf[ 11 ] );
      return 0;
    }
  angles[ 6 ] = buf[ 12 ] + (buf[ 13 ] << 8);
  printf( "angles: 3: %d; 4: %d; 5: %d; 6: %d (%d)\n",
	  angles[3], angles[4], angles[5], angles[6], read_angles_pending );
  read_angles_pending = 0;
  return 1;
}

/**********************************************************************/
/**********************************************************************/

int parse_set_angles( unsigned char buf[] )
{
  if ( buf[ 2 ] != 0x02 )
    {
      printf( "bad set_angles response : 0x%x\n", buf[ 2 ] );
      return 0;
    }
  return 1;
}

/**********************************************************************/
/**********************************************************************/
/**********************************************************************/
/**********************************************************************/

int parse_response( unsigned char buf[] )
{
  if ( buf[ 0 ] != 0x55 )
    {
      printf( "bad response: leading 0x55: 0x%x\n", buf[ 0 ] );
      return 0;
    }
  if ( buf[ 1 ] != 0x55 )
    {
      printf( "bad response: second 0x55: 0x%x\n", buf[ 1 ] );
      return 0;
    }
  switch( buf[ 3 ] )
    {
    case DATA_BATTERY_VOLTAGE:
      return parse_battery_voltage( buf );
    case DATA_MULT_SERVO_POS_READ:
      return parse_read_angles( buf );
    case CMD_MULT_SERVO_POS_WRITE:
      return parse_set_angles( buf );
    }
  printf( "Unknown response: %d 0x%x\n", buf[2], buf[3] );
  return 1;
}

/**********************************************************************/
/**********************************************************************/
// returns 1 if got response, 0 otherwise, no waiting or blocking

int check_for_xarm_response( int fd )
{
  unsigned char buf[80];
  int rdlen;

  rdlen = read(fd, buf, sizeof(buf) - 1);
  if (rdlen > 0)
    {
      buf[rdlen] = 0;
      /* display hex */
      unsigned char   *p;
      /*
      printf("Read %d:", rdlen);
      for (p = buf; rdlen-- > 0; p++)
	printf(" 0x%x", *p);
      printf("\n");
      */
      parse_response( buf );
      return 1;
    }
  else if (rdlen == -1 )
    return 0;
  else if ( rdlen < 0 )
    {
      printf( "Error from read: %d: %s\n", rdlen, strerror(errno) );
      close( fd );
      exit( -1 );
    }
}

/**********************************************************************/
// blocks, waiting for response from xarm to any sensor request

int wait_for_response( int fd )
{
  int count = 1;

  for ( count = 1; ; count++ )
    {
      if ( (count%1000) == 0 )
	printf( "wfr: Waiting for xarm response %d msec. This should not happen\n\n", count );
      if ( check_for_xarm_response( fd ) )
	break;
      usleep( 1000 ); // sleep for a millisecond
    }
  return count;
}

/**********************************************************************/
/**********************************************************************/

// set angles
#define N_SET_ANGLES 20
char SET_ANGLES[ N_SET_ANGLES ] = {
  0,
  0x55,
  0x55,
  0x11, // length
  CMD_SERVO_MOVE,
  0x4,
  0x0, // about 1000 ms.
  0x04,
  0x3,
  0x0,
  0x02,
  0x4,
  0x0,
  0x02,
  0x5,
  0x0,
  0x02,
  0x6,
  0x0,
  0x02
};

/**********************************************************************/
// goto a set of desired angles, no waiting

void set_angles( int fd, int *angles )
{
  int wlen;
  int i;

  for ( i = 3; i <= 6; i++ )
    {
      angles_desired[ i ] = angles[ i ];
      SET_ANGLES[ 9 + (i-3)*3 ] =  angles[ i ] & 0xff;
      SET_ANGLES[ 10 + (i-3)*3 ] =  (angles[ i ] >> 8) & 0xff;
    }
  wlen = write( fd, SET_ANGLES, N_SET_ANGLES );
  if ( wlen != N_SET_ANGLES )
    {
      printf( "Error from write: %d != %d; errno = %d\n", wlen,
	      N_SET_ANGLES, errno );
      exit( -1 );
    }
  // printf( "wrote: %d\n", wlen );
}

/**********************************************************************/
// goto a set of desired angles, and wait for move to complete.

int set_angles_and_wait( int fd, int *angles )
{
  int count = 1;

  set_angles( fd, angles );

  sleep( 2 );

  get_angles( fd );

  for ( count = 1; ; count++ )
    {
      if ( (count%1000) == 0 )
	printf( "saaw: Waiting for xarm response %d msec. This should not happen\n", count );
      if ( check_for_xarm_response( fd ) )
	break;
      usleep( 1000 ); // sleep for a millisecond
    }
  return count;
}

/**********************************************************************/
// open loop scailing

int scale(int desire_angle, int current_angle)
{
   int difference = 0;
   if(desire_angle < current_angle){
        difference = current_angle - desire_angle;
        if(difference < 2) return -1;
        else if(difference < 5) return -2;
        else if(difference < 7) return -3;
        else return -4;
   }else{
        difference = desire_angle - current_angle;
        if(difference < 2) return 1;
        else if(difference < 5) return 2;
        else if(difference < 7) return 3;
        else return 4;
   }

}

/**********************************************************************/
/**********************************************************************/

int main( int argc, char **argv)
{   
  int fd;
  char *device = "/dev/hidraw2";
  int wlen;
  int errors = 0;
  int angles_4 = 0;
  int angles_5 = 0;
  int angles_6 = 0;
  unsigned char buf[80];
  int count = 0;
  int rdlen;
  int flag_4 = -100;
  int flag_5 = -100;
  int flag_6 = -100;

  init_hidraw( device, &fd );

  get_battery_voltage( fd );

  wait_for_response( fd );

  /*
  while(1){
      get_angles( fd );

    wait_for_response( fd );
  }
  */
  

  
  int index = 0;
  int angles_d[ 10 ];
  
  for(index = 1;index <= argc-1;index=index+6){
     if(strcmp(argv[index+4],"-1") != 0){
        char final[500];
        strcpy(final,"./UscCmd --servo 3,");
        strcat(final, argv[index+4]);
        printf("%s\n", final);
        system(final);
     }
     if(strcmp(argv[index+3],"-1") != 0){
        char final[500];
        strcpy(final,"./UscCmd --servo 1,");
        strcat(final, argv[index+3]);
        printf("%s\n", final);
        system(final);
     }
     if(strcmp(argv[index+5],"-1") != 0){
        char final[500];
        strcpy(final,"./UscCmd --servo 5,");
        strcat(final, argv[index+5]);
        printf("%s\n", final);
        system(final);
     }
     if(strcmp(argv[index+2],"-1") != 0){
        printf("%d\n", atoi(argv[index+2]));
        angles_d[4] = atoi(argv[index+2]);
     }
     if(strcmp(argv[index+1],"-1") != 0){
        printf("%d\n", atoi(argv[index+1]));
        angles_d[5] = atoi(argv[index+1]);
     }
     if(strcmp(argv[index],"-1") != 0){
        printf("%d\n", atoi(argv[index]));
        angles_d[6] = atoi(argv[index]);
     }
     if(strcmp(argv[index+2],"-1") != 0 && strcmp(argv[index+1],"-1") != 0 && strcmp(argv[index],"-1") != 0){
       set_angles_and_wait(fd,angles_d);
       get_angles( fd );
       
       for ( count = 1; ; count++ ){
          if ( (count%1000) == 0 ) printf( "wfr: Waiting for xarm response %d msec. This should not happen\n\n", count );
          
          rdlen = read(fd, buf, sizeof(buf) - 1);
          
          if (rdlen > 0){
              angles_4= buf[ 6 ] + (buf[ 7 ] << 8);
              angles_5 = buf[ 9 ] + (buf[ 10 ] << 8);
              angles_6 = buf[ 12 ] + (buf[ 13 ] << 8);
              printf("4:%d,5:%d,6:%d\n",angles_4, angles_5, angles_6);
              break;
          }
          usleep( 1000 ); // sleep for a millisecond
       }
       
      flag_4 = abs(atoi(argv[index+2]) - angles_4);
      flag_5 = abs(atoi(argv[index+1]) - angles_5);
      flag_6 = abs(atoi(argv[index]) - angles_6);
      
      while(flag_4 > 1 || flag_5 > 1  || flag_6 > 1 ){
          if(abs(atoi(argv[index+2]) - angles_4) > 1){
             angles_d[4] += scale(atoi(argv[index+2]),angles_4);
          }
          if(abs(atoi(argv[index+1]) - angles_5) > 1){
             angles_d[5] += scale(atoi(argv[index+1]),angles_5);
          }
          if(abs(atoi(argv[index]) - angles_6) > 1){
             angles_d[6] += scale(atoi(argv[index]),angles_6);
          }

          set_angles_and_wait(fd,angles_d);
          get_angles( fd );
          for ( count = 1; ; count++ ){
            if ( (count%1000) == 0 ) printf( "wfr: Waiting for xarm response %d msec. This should not happen\n\n", count );
            
            rdlen = read(fd, buf, sizeof(buf) - 1);
            
            if (rdlen > 0){
                angles_4= buf[ 6 ] + (buf[ 7 ] << 8);
                angles_5 = buf[ 9 ] + (buf[ 10 ] << 8);
                angles_6 = buf[ 12 ] + (buf[ 13 ] << 8);
                printf("4:%d,5:%d,6:%d\n",angles_4, angles_5, angles_6);
                break;
            }
            usleep( 1000 ); // sleep for a millisecond
          }
          flag_4 = abs(atoi(argv[index+2]) - angles_4);
          flag_5 = abs(atoi(argv[index+1]) - angles_5);
          flag_6 = abs(atoi(argv[index]) - angles_6);
          
       }
     }
  }


  close(fd);
  
  return 0;
}

/**********************************************************************/
/**********************************************************************/
