/*
 * EventHandler.h
 *
 *  Created on: Jun 4, 2014
 *      Author: Mauricio Vanegas
 */

#ifndef EVENTHANDLER_H_
#define EVENTHANDLER_H_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <pthread.h>
#include <string.h>

struct TimerAllocator
{
	std::vector<void (*)()> tFunction;
	std::vector<int> tBaseTime;
	size_t length() const {return tFunction.size();};
};

class EventHandler {
public:
	EventHandler();
	EventHandler(int i);
	~EventHandler();
	void setTimerExec(void (*function)());
	void setTimerExec(void (*function)(), int baseTime);
	void setDown(void);
private:
	/* Variables */
	bool ClassExit;
	int baseTimeClass; /* Milliseconds */

	/* Timer definitions */
	TimerAllocator tPointers;
	pthread_t timer_handler, waitKey_handler;
	pthread_mutex_t timer_mutex;
	/*************/

	/* Functions */
	static void *StaticThreadLauncher(void* Param);
	void *TimerWorkaround(void);
	/*************/
};

#endif /* EVENTHANDLER_H_ */
