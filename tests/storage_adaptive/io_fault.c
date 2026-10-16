#define _GNU_SOURCE
#include <dlfcn.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h>
static const char *fd_path(int fd,char *buf,size_t cap) {
 char key[64];snprintf(key,sizeof key,"/proc/self/fd/%d",fd);
 ssize_t n=readlink(key,buf,cap-1);if(n<0){buf[0]=0;return buf;}buf[n]=0;return buf;
}
static int target(int fd) {char path[4096];return strstr(fd_path(fd,path,sizeof path),"codex_storage_fault_")!=NULL;}
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
 const char *mode=getenv("STORAGE_TEST_FAULT");
 if(mode && target(fd)) {
  char path[4096];fd_path(fd,path,sizeof path);
  // marker：只让「已完成日期」标记的临时文件确认失败（原行为）
  if(!strcmp(mode,"marker")) {if(strstr(path,"_completed_dates.atomic.tmp")){errno=EIO;return -1;}}
  // sync：每次确认都失败（原行为，用来验证最终仍必须报错退出）
  else if(!strcmp(mode,"sync")) {errno=EIO;return -1;}
  // sync_once：进程内第一次对目标「文件」的确认失败，之后一律放行。
  // 目录的确认不参与（目录确认不重试，让它失败只会掩盖被测的投影重试）。
  else if(!strcmp(mode,"sync_once")) {
   static int fired;struct stat st;
   if(!fired && fstat(fd,&st)==0 && !S_ISDIR(st.st_mode)){fired=1;errno=EIO;return -1;}
  }
  // sync_log：一律放行，每次确认在 stderr 留一行，便于数投影分了几段
  else if(!strcmp(mode,"sync_log")) {fprintf(stderr,"FAULT_FSYNC %s\n",path);}
 }
 return real_fn(fd);
}
