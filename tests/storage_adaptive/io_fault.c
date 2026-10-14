#define _GNU_SOURCE
#include <dlfcn.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
static int target(int fd) {
 char key[64],path[4096];snprintf(key,sizeof key,"/proc/self/fd/%d",fd);
 ssize_t n=readlink(key,path,sizeof path-1);if(n<0)return 0;path[n]=0;
 return strstr(path,"codex_storage_fault_")!=NULL;
}
static int read_fault(int fd,size_t n) {const char *mode=getenv("STORAGE_TEST_FAULT");return mode && !strcmp(mode,"read") && target(fd) && n>=4000;}
ssize_t pread64(int fd,void *buf,size_t n,off64_t off) {
 static ssize_t (*real_fn)(int,void*,size_t,off64_t);if(!real_fn)real_fn=dlsym(RTLD_NEXT,"pread64");
 if(read_fault(fd,n)){errno=EIO;return -1;}return real_fn(fd,buf,n,off);
}
ssize_t pread(int fd,void *buf,size_t n,off_t off) {
 static ssize_t (*real_fn)(int,void*,size_t,off_t);if(!real_fn)real_fn=dlsym(RTLD_NEXT,"pread");
 if(read_fault(fd,n)){errno=EIO;return -1;}return real_fn(fd,buf,n,off);
}
int fsync(int fd) {
 static int (*real_fn)(int);if(!real_fn)real_fn=dlsym(RTLD_NEXT,"fsync");
 const char *mode=getenv("STORAGE_TEST_FAULT");if(mode && !strcmp(mode,"marker") && target(fd)) {
 char key[64],path[4096];snprintf(key,sizeof key,"/proc/self/fd/%d",fd);ssize_t n=readlink(key,path,sizeof path-1);
 if(n>0){path[n]=0;if(strstr(path,"_completed_dates.atomic.tmp")){errno=EIO;return -1;}}
 }
 if(mode && !strcmp(mode,"sync") && target(fd)){errno=EIO;return -1;}return real_fn(fd);
}
