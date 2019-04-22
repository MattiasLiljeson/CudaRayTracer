#include "PerfTimer.h"

PerfTimer::PerfTimer( string p_name, bool p_doLog )
{
	name = p_name;
	doLog = p_doLog;
	lastTime = 0.0f;
	timer.reset();
}

void PerfTimer::start()
{
	timer.reset();
	timer.start();
}

void PerfTimer::stop()
{
	static const float millisFac = 1000.0f;
	timer.stop();
	lastTime = (float)timer.getGameTime() * millisFac;

	if( doLog ){
		log.push_back( lastTime );
	}
}

void PerfTimer::reset()
{
	log.clear();
}

double PerfTimer::calcStdDev() {
	double tot = 0;
	//double times[VALUE_CNT];
	for ( unsigned i=0; i<log.size(); i++ ) {
		//double conv = cyclesToSeconds(m_samples[i]);
		//times[i] = conv;
		tot += log[i];
	}
	double avg = tot/log.size();

	//double diffsSquared[VALUE_CNT];
	vector<double> diffsSquared;
	for ( unsigned i=0; i<log.size(); i++ ) {
		double diff = log[i] - avg;
		diffsSquared.push_back( diff*diff );
	}

	double diffsSquaredSum = 0; 
	for ( unsigned i=0; i<diffsSquared.size(); i++ ) {
		diffsSquaredSum += diffsSquared[i];
	}
	double stdDev = diffsSquaredSum/ diffsSquared.size();
	stdDev = sqrt(stdDev);

	return stdDev;
} 

double PerfTimer::avg() {
	double tot = 0;
	//double times[VALUE_CNT];
	for ( unsigned i=0; i<log.size(); i++ ) {
		//double conv = cyclesToSeconds(m_samples[i]);
		tot += log[i];
	}
	double avg = tot/log.size();
	return avg;
}

template <typename T>
int compare( const void * a, const void * b )
{
	if ( *(T*)a <  *(T*)b ) return -1;
	if ( *(T*)a == *(T*)b ) return 0;
	if ( *(T*)a >  *(T*)b ) return 1;
	else return 0;
}

double PerfTimer::median()
{
	//int tmp[VALUE_CNT];
	vector<double> tmps;
	for ( unsigned i=0; i<log.size(); i++ ) {
		tmps.push_back(log[i]);
	}
	qsort(&tmps[0], log.size(), sizeof(double), compare<double>);

	//int zeroCnt = VALUE_CNT-m_adds;
	//int medianIdx = zeroCnt + m_adds/2;

	//int medianAsInt = tmps[log.size()/2];
	//double medianAsDouble = cyclesToSeconds(medianAsInt);
	double medianAsDouble = tmps[log.size()/2];
	return medianAsDouble;
}


//double PerfTimer::cyclesToSeconds( int clocks )
//{
//	LARGE_INTEGER clocksToSecsFac;
//	QueryPerformanceFrequency(&clocksToSecsFac);
//	double fac = (double)clocksToSecsFac.LowPart;
//	fac = 3600000000; // clock speed on i7 2600k
//	fac = 1000.0; // millis to secs
//	return clocks / fac;
//}