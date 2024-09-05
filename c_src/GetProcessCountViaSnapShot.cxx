#include "intelligence.h"
extern "C" {
    BOOL GetProcessCountViaSnapShot(OUT DWORD* dwProcessCount) {

        PROCESSENTRY32  ProcEntry						= { .dwSize = sizeof(PROCESSENTRY32) };
        HANDLE			hSnapShot						= INVALID_HANDLE_VALUE;
        DWORD           dwProcCount                     = 0x0;

        if (!dwProcessCount){
            return FALSE;
        }

        if ((hSnapShot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0x0)) == INVALID_HANDLE_VALUE) {
            PRINT("[!] CreateToolhelp32Snapshot Failed With Error: %ld \n", GetLastError());
            return FALSE;
        }

        if (!Process32First(hSnapShot, &ProcEntry)) {
            PRINT("[!] Process32First Failed With Error: %ld \n", GetLastError());
            goto _END_OF_FUNC;
        }

        do {

            dwProcCount++;

        } while (Process32Next(hSnapShot, &ProcEntry));

        *dwProcessCount = dwProcCount;

    _END_OF_FUNC:
        if (hSnapShot != INVALID_HANDLE_VALUE){
            CloseHandle(hSnapShot);
        }

        return (*dwProcessCount) ? TRUE : FALSE;
    }
}