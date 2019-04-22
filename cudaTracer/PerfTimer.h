#pragma once

#include <string>
#include <vector>
#include "Timer.h"

using namespace std;

struct PerfTimer
{
	static const int LOGGING_DISABLED = -1;
	Timer timer;
	string name;
	bool doLog;
	vector<double> log;
	float lastTime;

	PerfTimer( string p_name, bool p_doLog );
	void start();
	void stop();
	void reset();

	double calcStdDev();
	double avg();
	double median();

	//double cyclesToSeconds( int clocks );
};