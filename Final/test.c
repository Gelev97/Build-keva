#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>

int main( int argc, char **argv)
{
  int index = 0;
  int angles_d[ 10 ];
  for(index = 1;index <= argc-1;index=index+6){
     if(strcmp(argv[index],"-1") != 0){
        angles_d[6] = atoi(argv[index]);
        printf("%d\n", angles_d[6]);
     }
     if(strcmp(argv[index+1],"-1") != 0){
        angles_d[5] = atoi(argv[index+1]);
        printf("%d\n", angles_d[5]);
     }
     if(strcmp(argv[index+2],"-1") != 0){
        angles_d[4] = atoi(argv[index+2]);
        printf("%d\n", angles_d[4]);
     }
     if(strcmp(argv[index+3],"-1") != 0){
        char final[500];
        strcpy(final,"./UscCmd --servo 1,");
        strcat(final, argv[index+3]);
        printf("%s\n", final);
     }
     if(strcmp(argv[index+4],"-1") != 0){
        char final[500];
        strcpy(final,"./UscCmd --servo 3,");
        strcat(final, argv[index+4]);
        printf("%s\n", final);
     }
     if(strcmp(argv[index+5],"-1") != 0){
        char final[500];
        strcpy(final,"./UscCmd --servo 5,");
        strcat(final, argv[index+5]);
        printf("%s\n", final);
     }
  }

  return 0;
}