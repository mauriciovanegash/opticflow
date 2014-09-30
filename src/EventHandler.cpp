/*
 * EventHandler.cpp
 *
 *  Created on: Jun 4, 2014
 *      Author: Mauricio Vanegas
 */

#include "EventHandler.h"

EventHandler::EventHandler() : ClassExit(false), baseTimeClass(10) {
	if(pthread_mutex_init( &timer_mutex, NULL )!=0)
	{
		printf( "TimerWorkaround() thread launching:\n" );
		printf(	"Timer mutex_thread initialisation failed.\n" );
		printf( "Exiting!!!\n" );
		exit( 1 );
	}

	if(pthread_create(&timer_handler, NULL, &StaticThreadLauncher, this)!=0)
	{
		printf( "TimerWorkaround() thread launching:\n" );
		printf(	"Thread creation failed.\n" );
		printf( "Exiting!!!\n" );
		exit( 1 );
	}
}

EventHandler::EventHandler(int i) : ClassExit(false), baseTimeClass(i)
{
	if(pthread_mutex_init( &timer_mutex, NULL )!=0)
	{
		printf( "TimerWorkaround() thread launching:\n" );
		printf(	"Timer mutex_thread initialisation failed.\n" );
		printf( "Exiting!!!\n" );
		exit( 1 );
	}

	if(pthread_create(&timer_handler, NULL, &StaticThreadLauncher, this)!=0)
	{
		printf( "TimerWorkaround() thread launching:\n" );
		printf(	"Thread creation failed.\n" );
		printf( "Exiting!!!\n" );
		exit( 1 );
	}
}

EventHandler::~EventHandler()
{
	ClassExit = true;
	pthread_join(timer_handler,NULL);
}

void *EventHandler::StaticThreadLauncher(void* Param)
{
	printf("Starting ...\n");
	return ((EventHandler *)Param)->TimerWorkaround();
}

void *EventHandler::TimerWorkaround(void)
{
	int milisec = 1; // length of time to sleep, in milliseconds...
	struct timespec mTime = {0};
	mTime.tv_sec = 0;
	mTime.tv_nsec = milisec * 1000000L;
	int milisecondTimer = 0;

	while(!ClassExit)
	{
		pthread_mutex_lock( &timer_mutex );
		if(tPointers.length()>0)
		{
			std::vector<int>::iterator itBaseTime = tPointers.tBaseTime.begin();
			for(std::vector<void (*)()>::iterator itFunc = tPointers.tFunction.begin(); itFunc!=tPointers.tFunction.end();++itFunc)
			{
				if((milisecondTimer%*itBaseTime)==0)
					(*itFunc)();
				++itBaseTime;
			}
		}
		pthread_mutex_unlock( &timer_mutex );
		nanosleep(&mTime, (struct timespec *)NULL);
		++milisecondTimer;
	}
	pthread_exit(NULL);
	return this;
}

void EventHandler::setTimerExec(void (*function)())
{
	printf("Function allocated!!\n");
	pthread_mutex_lock( &timer_mutex );
	tPointers.tFunction.push_back(function);
	tPointers.tBaseTime.push_back(baseTimeClass);
	pthread_mutex_unlock( &timer_mutex );
}

void EventHandler::setTimerExec(void (*function)(),int baseTime)
{
	printf("Function allocated!!\n");
	pthread_mutex_lock( &timer_mutex );
	tPointers.tFunction.push_back(function);
	tPointers.tBaseTime.push_back(baseTime);
	pthread_mutex_unlock( &timer_mutex );
}

void EventHandler::setDown(void)
{
	ClassExit = true;
}
