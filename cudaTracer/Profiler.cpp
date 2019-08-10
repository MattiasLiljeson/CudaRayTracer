#include <ctime>
#include <fstream>

#include <filesystem>
#include "DebugGUI.h"
#include "Profiler.h"

//=========================================================================
// Public functions
//=========================================================================

Profiler* Profiler::instance() {
    // Instantiated on first use. Guaranteed to be destroyed.
    static Profiler instance;
    return &instance;
}

Profiler::~Profiler(void) {
    logTimersToFile(true, "viaDestructor");
    for (auto it = m_timers.begin(); it != m_timers.end(); ++it) {
        delete it->second;
    }
}

void Profiler::addPerfTimer(const string& p_timer, const string& p_group,
                            bool p_doLog) {
    // Don't overwrite existing timers. TODO: indicate? exception?
    if (getTimer(p_timer) == nullptr) {
        PerfTimer* perfTimer = new PerfTimer(p_timer, p_doLog);
        m_timers[p_timer] = perfTimer;

        string options = "";
        if (p_group != "") {
            options = " group=" + p_group + " ";
        }

        DebugGUI* dg = ServiceRegistry::instance().get<DebugGUI>();
        dg->addVar(m_bar, DebugGUI::DG_FLOAT, DebugGUI::READ_ONLY,
                   perfTimer->name, &(perfTimer->lastTime), options);
    }
}

void Profiler::start(const string& p_timer, const string& p_group,
                     bool p_doLog) {
#ifdef USE_PROFILING
    addPerfTimer(p_timer, p_group, p_doLog);  // Add if not existing already
    m_timers[p_timer]->start();
#endif
}

void Profiler::stop(const string& p_timer, const string& p_group,
                    bool p_doLog) {
#ifdef USE_PROFILING
    addPerfTimer(p_timer, p_group, p_doLog);  // Add if not existing already
    m_timers[p_timer]->stop();
#endif
}

void Profiler::logTimersToFile(bool p_timeStamp, string tag) {
    vector<PerfTimer*> timersToLog;
    addLogTimersToVector(timersToLog);

    if (!timersToLog.empty()) {
        {
            string fname = createFname(p_timeStamp, tag);
            bool append = std::filesystem::exists(fname);
            ofstream logFile;
            if (createLogFile(fname, logFile)) {
                logTimers(timersToLog, logFile, append);
                logFile.close();
            }
        }
        {
            string fname = createFname(p_timeStamp, tag + string("-STATS"));
            bool append = std::filesystem::exists(fname);
            ofstream statsFile;
            if (createLogFile(fname, statsFile)) {
                logTimerStats(timersToLog, statsFile, append);
                statsFile.close();
            }
        }
        for (int i = 0; i < timersToLog.size(); i++) {
            timersToLog[i]->reset();
        }
    }
}

//=========================================================================
// Private functions
//=========================================================================

Profiler::Profiler(void) { m_bar = "Performance"; }

PerfTimer* Profiler::getTimer(string p_timer) {
    if (m_timers.find(p_timer) == m_timers.end()) {
        return nullptr;
    } else {
        return m_timers[p_timer];
    }
}

std::string Profiler::createFname(bool p_timeStamp, string tag) {
    long timeNow = (long)time(nullptr);
    int charsWritten = 0;
    const int BUFF_SIZE = 128;
    char nameBuf[BUFF_SIZE];
    if (p_timeStamp) {
        charsWritten =
            sprintf_s(nameBuf, BUFF_SIZE, "../profileLogs/log%d-%s.txt",
                      timeNow, tag.c_str());
    } else {
        charsWritten =
            sprintf_s(nameBuf, BUFF_SIZE, "../profileLogs/%s.txt", tag.c_str());
    }
    return std::string(nameBuf);
}

bool Profiler::createLogFile(string p_fileName, ofstream& p_file) {
    bool fileOk = false;
    p_file.close();
    p_file.open(p_fileName, std::ofstream::out | std::ofstream::app);
    if (p_file.good()) {
        fileOk = true;
    }

    return fileOk;
}

void Profiler::logTimer(const PerfTimer* p_perfTimer, ostream& p_file) {
    p_file << p_perfTimer->name;
    for (unsigned int i = 0; i < p_perfTimer->log.size(); i++) {
        p_file << '\t' << p_perfTimer->log[i];
    }
}

void Profiler::logTimers(const vector<PerfTimer*>& p_perfTimers,
                         ostream& p_file, bool append) {
    if (!append) {
        p_file << "step";
        for (unsigned int stepIdx = 0; stepIdx < p_perfTimers.size();
             stepIdx++) {
            p_file << '\t' << p_perfTimers[stepIdx]->name;
        }
    }

    if (p_perfTimers.size() > 0) {
        for (unsigned int stepIdx = 0; stepIdx < p_perfTimers[0]->log.size();
             stepIdx++) {
            p_file << '\n' << stepIdx;
            for (unsigned int timerIdx = 0; timerIdx < p_perfTimers.size();
                 timerIdx++) {
                p_file << '\t' << p_perfTimers[timerIdx]->log[stepIdx];
            }
        }
    }
}

void Profiler::logTimerStats(const vector<PerfTimer*>& p_perfTimers,
                             ostream& p_file, bool append) {
    if (!append) {
        p_file << "Timer \t avg \t median \t stdDev \n";
    }
    for (unsigned int timerIdx = 0; timerIdx < p_perfTimers.size();
         timerIdx++) {
        p_file << p_perfTimers[timerIdx]->name;
        p_file << '\t' << p_perfTimers[timerIdx]->avg();
        p_file << '\t' << p_perfTimers[timerIdx]->median();
        p_file << '\t' << p_perfTimers[timerIdx]->calcStdDev();
        p_file << '\n';
    }
}

void Profiler::addLogTimersToVector(vector<PerfTimer*>& timersToLog) {
    for (auto it = m_timers.begin(); it != m_timers.end(); ++it) {
        if (it->second->log.size() > 0) {
            timersToLog.push_back(it->second);
        }
    }
}