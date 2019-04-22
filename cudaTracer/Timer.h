//Code from Luna. MUST BE REPLACED!
#ifndef TIMER_H
#define TIMER_H

#include <Windows.h>

class Timer
{
private:
	double mSecondsPerCount;
	double mDeltaTime;

	__int64 mBaseTime;
	__int64 mPausedTime;
	__int64 mStopTime;
	__int64 mPrevTime;
	__int64 mCurrTime;

	bool mStopped;

public:
	Timer();

	float getGameTime()const;  // in seconds
	float getDt()const; // in seconds

	void reset(); // Call before message loop.
	void start(); // Call when unpaused.
	void stop();  // Call when paused.
	void tick();  // Call every frame.
};

#endif //TIMER_H