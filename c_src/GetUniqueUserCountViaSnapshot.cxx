#include "intelligence.h"

#define MAX_NAME 256
#define HASH_TABLE_SIZE 1024

// based on: https://stackoverflow.com/questions/2686096/c-get-username-from-process

// Simple hash table implementation to track unique user names
typedef struct HashEntry {
    char user[MAX_NAME * 2];
    struct HashEntry* next;
} HashEntry;

typedef struct HashTable {
    HashEntry* entries[HASH_TABLE_SIZE];
} HashTable;

unsigned int hash(const char* str) {
    unsigned int hash = 0;
    while (*str) {
        hash = (hash << 5) + hash + *str++;
    }
    return hash % HASH_TABLE_SIZE;
}

void insert(HashTable* table, const char* user) {
    unsigned int index = hash(user);
    HashEntry* entry   = table->entries[index];
    
    while (entry != NULL) {
        
        if (strcmp(entry->user, user) == 0) {
            return; // User already in table
        }
        entry = entry->next;
    }

    entry = (HashEntry*)malloc(sizeof(HashEntry));
    strcpy(entry->user, user);
    entry->next = table->entries[index];
    table->entries[index] = entry;
}

void clearTable(HashTable* table) {
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        
        HashEntry* entry = table->entries[i];
        
        while (entry != NULL) {
            HashEntry* prev = entry;
            entry = entry->next;
            free(prev);
        }
        table->entries[i] = NULL;
    }
}

BOOL GetLogonFromToken(HANDLE hToken, char* strUser, char* strDomain) {
    DWORD dwSize     =  MAX_NAME;
    BOOL bSuccess    =  FALSE;
    DWORD dwLength   =  0;
    PTOKEN_USER ptu  =  NULL;

    // Verify the parameter passed in is not NULL.
    if (NULL == hToken) {
        goto _END_OF_FUNC;
    }

    if (!GetTokenInformation(
            hToken,      // handle to the access token
            TokenUser,   // get information about the token's groups
            (LPVOID)ptu, // pointer to PTOKEN_USER buffer
            0,           // size of buffer
            &dwLength    // receives required buffer size
        )) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            PRINT("GetTokenInformation Error %ld\n", GetLastError());
            goto _END_OF_FUNC;
        }

        ptu = (PTOKEN_USER)HeapAlloc(GetProcessHeap(),
            HEAP_ZERO_MEMORY, dwLength);

        if (ptu == NULL) {
            goto _END_OF_FUNC;
        }
    }

    if (!GetTokenInformation(
            hToken,      // handle to the access token
            TokenUser,   // get information about the token's groups
            (LPVOID)ptu, // pointer to PTOKEN_USER buffer
            dwLength,    // size of buffer
            &dwLength    // receives required buffer size
        )) {
        goto _END_OF_FUNC;
    }

    SID_NAME_USE SidType;
    char lpName[MAX_NAME];
    char lpDomain[MAX_NAME];

    if (!LookupAccountSid(NULL, ptu->User.Sid, lpName, &dwSize, lpDomain, &dwSize, &SidType)) {
        DWORD dwResult = GetLastError();
        if (dwResult == ERROR_NONE_MAPPED) {
            strcpy(lpName, "NONE_MAPPED");
        } else {
            PRINT("LookupAccountSid Error %ld\n", GetLastError());
        }
    } else {
        strcpy(strUser, lpName);
        strcpy(strDomain, lpDomain);
        bSuccess = TRUE;
    }

_END_OF_FUNC:

    if (ptu != NULL) {
        HeapFree(GetProcessHeap(), 0, (LPVOID)ptu);
    }
    return bSuccess;
}

BOOL GetUserFromProcess(const DWORD procId, char* strUser, char* strDomain, char* placeHolder, DWORD* last_error) {
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, procId);
    HANDLE hToken   = NULL;
	DWORD  error    = GetLastError();

    if (hProcess == NULL) {
	
		if(error != *last_error){
			placeHolder[0] += 1;
			*last_error = error;
		}
		
        return FALSE;
    }
    
    if (!OpenProcessToken(hProcess, TOKEN_QUERY, &hToken)) {
		error    = GetLastError();
		
		if(error != *last_error){
			placeHolder[0] += 1;
			*last_error = error;
		}
		
        CloseHandle(hProcess);
        return FALSE;
    }
    
    BOOL bres = GetLogonFromToken(hToken, strUser, strDomain);

    CloseHandle(hToken);
    CloseHandle(hProcess);
    return bres ? TRUE : FALSE;
}

extern "C" {
    BOOL GetUniqueUserCountViaSnapshot(OUT DWORD* dwUserCount) {
        PROCESSENTRY32 ProcEntry  =  { .dwSize = sizeof(PROCESSENTRY32) };
        HANDLE hSnapShot          =  INVALID_HANDLE_VALUE;
        char PlaceHolder[]        =  "0_PlaceHolder";
        HashTable uniqueUsers     =  { 0 }; // Initialize hash table
        DWORD lastErrorCode       =  0x0;   // Lazy unique error code check

        if (!dwUserCount) {
            return FALSE;
        }

        if ((hSnapShot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0x0)) == INVALID_HANDLE_VALUE) {
            PRINT("[!] CreateToolhelp32Snapshot Failed With Error: %ld \n", GetLastError());
            return FALSE;
        }

        if (!Process32First(hSnapShot, &ProcEntry)) {
            PRINT("[!] Process32First Failed With Error: %ld \n", GetLastError());
            CloseHandle(hSnapShot);
            return FALSE;
        }

        do {
            char strUser[MAX_NAME], strDomain[MAX_NAME];
            if (GetUserFromProcess(ProcEntry.th32ProcessID, strUser, strDomain, PlaceHolder, &lastErrorCode)) {
                insert(&uniqueUsers, strUser);
            }else{
                insert(&uniqueUsers, PlaceHolder);
            }

        } while (Process32Next(hSnapShot, &ProcEntry));

        *dwUserCount = 0;

        for (int i = 0; i < HASH_TABLE_SIZE; i++) {
            HashEntry* entry = uniqueUsers.entries[i];
            while (entry != NULL) {
                (*dwUserCount)++;
                entry = entry->next;
            }
        }

        if (hSnapShot != INVALID_HANDLE_VALUE) {
            CloseHandle(hSnapShot);
        }

        clearTable(&uniqueUsers);

        return (*dwUserCount) ? TRUE : FALSE;
    }
}
