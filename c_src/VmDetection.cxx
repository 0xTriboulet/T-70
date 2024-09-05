#include "intelligence.h"
extern "C" {
    BOOL VmDetection(float process_count_per_user){

        // Conditional extracted from DecisionTreeClassifier learnings
        if ((process_count_per_user > 75.3) || (process_count_per_user > 61.45 && process_count_per_user <= 69.3)){

            PRINT("[i] Running on bare metal machine!\n");
            return TRUE;

        }

        return FALSE;

    }
}