#pragma once

#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "PerfTimer.h"

using namespace std;

class Profiler {
   public:
    static Profiler* instance();
    virtual ~Profiler(void);

    void addPerfTimer(const string& p_timer, const string& p_group,
                      bool p_doLog);
    void start(const string& p_timer, const string& p_group = "",
               bool p_doLog = false);
    void stop(const string& p_timer, const string& p_group = "",
              bool p_doLog = false);
    void logTimersToFile(bool p_timeStamp, string tag);

   private:
    Profiler();
    Profiler(Profiler const&);        // Don't Implement
    void operator=(Profiler const&);  // Don't implement

    PerfTimer* getTimer(string p_timer);
    std::string createFname(bool p_timeStamp, string tag);
    bool createLogFile(string p_fileName, ofstream& p_file);
    void logTimer(const PerfTimer* p_perfTimer, ostream& p_file);
    void logTimers(const vector<PerfTimer*>& p_perfTimers, ostream& p_file,
                   bool append);
    void logTimerStats(const vector<PerfTimer*>& p_perfTimers, ostream& p_file,
                       bool append);
    void addLogTimersToVector(vector<PerfTimer*>& timersToLog);

   private:
    map<string, PerfTimer*> m_timers;
    string m_bar;
};