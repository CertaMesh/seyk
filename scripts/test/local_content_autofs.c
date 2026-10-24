// Test-only autofs v5 daemon. Run in a disposable privileged Linux container:
// cc -Wall -Wextra -Werror scripts/test/local_content_autofs.c -o /tmp/autofs
// /tmp/autofs <seyk-core unit-test binary> [ancestor]
// No daemon package, network, backing service, or host mount is needed.
#define _GNU_SOURCE
#include <fcntl.h>
#include <linux/auto_fs.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static void check(int result, const char *operation) {
    if (result < 0) { perror(operation); exit(1); }
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) return 2;
    alarm(60);
    check(unshare(CLONE_NEWNS), "unshare");
    check(mount(NULL, "/", NULL, MS_REC | MS_PRIVATE, NULL), "private mounts");
    char base[] = "/tmp/seyk-autofs-XXXXXX";
    if (!mkdtemp(base)) { perror("mkdtemp"); return 1; }
    char trigger[256], selected[272], note[288], options[128];
    snprintf(trigger, sizeof(trigger), "%s/collection", base);
    snprintf(selected, sizeof(selected), "%s/selected", trigger);
    check(mkdir(trigger, 0700), "mkdir trigger");
    int pipefd[2];
    check(pipe(pipefd), "pipe");
    snprintf(options, sizeof(options), "fd=%d,pgrp=%d,minproto=5,maxproto=5,direct",
             pipefd[1], (int)getpgrp());
    check(mount("seyk-test", trigger, "autofs", 0, options), "mount autofs");
    int control = open(trigger, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    check(control, "autofs control");
    check(setenv("SEYK_AUTOFS_TRIGGER", trigger, 1), "setenv");
    check(setenv("SEYK_AUTOFS_ANCESTOR", argc == 3 ? "1" : "0", 1), "setenv");
    fflush(NULL);
    pid_t child = fork();
    check(child, "fork");
    if (child == 0) {
        // Requests from the daemon's own process group bypass automounts.
        check(setpgid(0, 0), "child pgrp");
        close(pipefd[0]);
        close(pipefd[1]);
        execl(argv[1], argv[1], "local_content_real_cold_autofs", "--ignored",
              "--nocapture", "--test-threads=1", NULL);
        perror("exec test");
        _exit(127);
    }
    int status, requests = 0;
    for (;;) {
        pid_t done = waitpid(child, &status, WNOHANG);
        check(done, "waitpid");
        if (done == child) break;
        struct pollfd pfd = { .fd = pipefd[0], .events = POLLIN };
        check(poll(&pfd, 1, 100), "poll");
        if (!(pfd.revents & POLLIN)) continue;
        struct autofs_v5_packet packet;
        ssize_t bytes = read(pipefd[0], &packet, sizeof(packet));
        if (bytes != sizeof(packet) || packet.hdr.proto_version != 5 ||
            packet.hdr.type != autofs_ptype_missing_direct) {
            fprintf(stderr, "unexpected autofs packet (%zd bytes)\n", bytes);
            kill(child, SIGKILL);
            return 1;
        }
        check(mount("seyk-sentinel", trigger, "tmpfs", 0, "size=1m"), "mount backing");
        check(mkdir(selected, 0700), "mkdir selected");
        snprintf(note, sizeof(note), "%s/note.rs", argc == 3 ? selected : trigger);
        FILE *file = fopen(note, "w");
        if (!file) { perror("sentinel"); return 1; }
        fputs("public replacement sentinel", file);
        check(fclose(file), "close sentinel");
        printf("autofs v5 request %d: mounted local tmpfs\n", ++requests);
        fflush(stdout);
        check(ioctl(control, AUTOFS_IOC_READY, packet.wait_queue_token), "AUTOFS_IOC_READY");
    }
    ioctl(control, AUTOFS_IOC_CATATONIC, 0);
    umount2(trigger, MNT_DETACH);
    umount2(trigger, MNT_DETACH);
    close(control);
    close(pipefd[0]);
    close(pipefd[1]);
    rmdir(trigger);
    rmdir(base);
    return requests > 0 && WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
