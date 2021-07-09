#include <stdlib.h>
#include <sys/resource.h>

#include <iostream>

using namespace std;

int main() {
    struct rlimit rlim;
    cout << "RLIM_INFINITY" << RLIM_INFINITY << endl;
    // int res = getrlimit(RLIMIT_NOFILE, &rlim);
    int res = getrlimit(RLIMIT_DATA, &rlim);
    if (!res) {
        cout << "soft:" << rlim.rlim_cur << ", hard: " << rlim.rlim_max << endl;
    } else {
        cout << "error code: " << res << endl;
    }

    FILE *f;
    f = fopen("/proc/sys/fs/file-max", "r");
    long fsmax;
    fscanf(f, "%ld", &fsmax);
    fclose(f);
    cout << "fsmax: " << fsmax << endl;

    return 0;
}
