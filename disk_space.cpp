#include <sys/statvfs.h>

#include <iostream>

int main() {
    struct statvfs f;
    statvfs("/home/krisloveless", &f);
    unsigned long available = f.f_bavail * f.f_frsize / 1024;
    std::cout << available << "K" << std::endl;
    return 0;
}
