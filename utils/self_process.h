// self_process.hpp — spawn a copy of the currently running executable, portably.
#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#elif defined(__APPLE__)
  #include <libproc.h>
  #include <unistd.h>
  #include <sys/wait.h>
  #include <spawn.h>
  extern char **environ;
#else // Linux / other POSIX
  #include <unistd.h>
  #include <sys/wait.h>
  #include <spawn.h>
  #include <climits>
  extern char **environ;
  #if defined(__FreeBSD__)
    #include <sys/types.h>
    #include <sys/sysctl.h>
  #endif
#endif

namespace selfproc {

// Returns the absolute, resolved path to the currently running executable.
inline std::filesystem::path executable_path() {
#if defined(_WIN32)
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        DWORD len = GetModuleFileNameW(nullptr, buf.data(), (DWORD)buf.size());
        if (len == 0) throw std::runtime_error("GetModuleFileNameW failed");
        if (len < buf.size() - 1) { buf.resize(len); break; }
        buf.resize(buf.size() * 2);
    }
    return std::filesystem::path(buf);

#elif defined(__APPLE__)
    std::array<char, PROC_PIDPATHINFO_MAXSIZE> buffer{};
    pid_t pid = getpid();
    
    int bytes = proc_pidpath(pid, buffer.data(), buffer.size());
    if (bytes > 0) {
        return std::filesystem::path(buffer.data());
    }
    
    return {}; // Return empty path on failure
#elif defined(__linux__)
    std::error_code ec;
    auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) throw std::runtime_error("reading /proc/self/exe failed: " + ec.message());
    return p;

#elif defined(__FreeBSD__)
    int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
    char buf[PATH_MAX];
    size_t len = sizeof(buf);
    if (sysctl(mib, 4, buf, &len, nullptr, 0) != 0)
        throw std::runtime_error("sysctl KERN_PROC_PATHNAME failed");
    return std::filesystem::path(std::string(buf, len ? len - 1 : 0));

#else
    #error "executable_path() not implemented for this platform"
#endif
}

#if defined(_WIN32)

struct Process {
    PROCESS_INFORMATION pi{};

    int wait() {
        WaitForSingleObject(pi.hProcess, INFINITE);
        DWORD code = 0;
        GetExitCodeProcess(pi.hProcess, &code);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        return static_cast<int>(code);
    }
};

namespace detail {
inline std::wstring to_wide(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
    return w;
}

// Correct Win32 argv-style quoting (handles embedded quotes/backslashes).
inline std::wstring quote_arg(const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos)
        return a;
    std::wstring out = L"\"";
    for (auto it = a.begin(); ; ++it) {
        size_t backslashes = 0;
        while (it != a.end() && *it == L'\\') { ++backslashes; ++it; }
        if (it == a.end()) {
            out.append(backslashes * 2, L'\\');
            break;
        } else if (*it == L'"') {
            out.append(backslashes * 2 + 1, L'\\');
            out.push_back(L'"');
        } else {
            out.append(backslashes, L'\\');
            out.push_back(*it);
        }
    }
    out.push_back(L'"');
    return out;
}
} // namespace detail

inline Process spawn(const std::filesystem::path& exe,
                      const std::vector<std::string>& args) {
    std::wstring cmdline = detail::quote_arg(exe.wstring());
    for (auto& a : args) {
        cmdline.push_back(L' ');
        cmdline += detail::quote_arg(detail::to_wide(a));
    }

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    std::vector<wchar_t> mutable_cmd(cmdline.begin(), cmdline.end());
    mutable_cmd.push_back(L'\0');

    BOOL ok = CreateProcessW(
        exe.wstring().c_str(),   // application name (avoids PATH search ambiguity)
        mutable_cmd.data(),      // command line, must be a mutable buffer
        nullptr, nullptr,
        TRUE,
        0, nullptr, nullptr,
        &si, &pi);

    if (!ok)
        throw std::runtime_error("CreateProcessW failed, error " + std::to_string(GetLastError()));

    Process p;
    p.pi = pi;
    return p;
}

#else // POSIX (Linux, macOS, FreeBSD, ...)

struct Process {
    pid_t pid = -1;

    int wait() {
        int status = 0;
        if (waitpid(pid, &status, 0) < 0)
            throw std::runtime_error("waitpid failed");
        if (WIFEXITED(status)) return WEXITSTATUS(status);
        if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
        return -1;
    }
};

inline Process spawn(const std::filesystem::path& exe,
                      const std::vector<std::string>& args) {

    std::vector<std::string> argv_storage;
    argv_storage.push_back(exe.string());
    for (auto& a : args) argv_storage.push_back(a);

    std::vector<char*> argv;
    for (auto& s : argv_storage) argv.push_back(s.data());
    argv.push_back(nullptr);

    pid_t pid;
    int rc = posix_spawn(&pid, exe.c_str(), nullptr, nullptr, argv.data(), environ);
    if (rc != 0)
        throw std::runtime_error(std::string("posix_spawn failed: ") + std::strerror(rc));

    Process p;
    p.pid = pid;
    return p;
}

#endif

// Convenience: spawn a fresh copy of the currently running executable with the given args.
inline Process spawn_self(const std::vector<std::string>& args) {
    return spawn(executable_path(), args);
}

} // namespace selfproc