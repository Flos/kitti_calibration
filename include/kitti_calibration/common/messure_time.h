/*
 * Messure_time.h
 *
 *  Created on: 19.02.2015
 *      Author: fnolden
 */

#include <ctime>

#ifndef INCLUDE_KITTI_CALIBRATION_COMMON_MESSURE_TIME_H_
#define INCLUDE_KITTI_CALIBRATION_COMMON_MESSURE_TIME_H_

const std::string spacer = "\t";

std::string time_string(clock_t start, clock_t end, std::string name, int iterations, int size1, int size2, int score=0, bool description = false){
	std::stringstream ss;
	if(description){
		ss << "name" << spacer;
		ss << "start" << spacer << "stop" << spacer;
		ss << "duration" << spacer;
		ss << "iterations" << spacer << "time/iteration" << spacer;
		ss << "size1"	<< spacer << "size2" << spacer;
		ss << "score";

		ss << std::endl;
	}

	ss << name << spacer;
	ss << ((float)start)/CLOCKS_PER_SEC  << spacer << ((float)end)/CLOCKS_PER_SEC << spacer;
	ss << ((double)end - start)/CLOCKS_PER_SEC<< spacer;
	ss << iterations << spacer << ((double(end-start))/CLOCKS_PER_SEC)/iterations << spacer;
	ss << size1 <<spacer << size2 << spacer;
	ss << score;
	ss << std::endl;

	return ss.str();
}

std::string time_diff(clock_t start, clock_t end){
	std::stringstream ss;
	ss << ((double)end - start)/CLOCKS_PER_SEC;
	return ss.str();
}

std::string datetime(){
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	time (&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer,80,"%d-%m-%Y %I:%M:%S",timeinfo);
	std::string str(buffer);

	return str;
}

#endif /* INCLUDE_KITTI_CALIBRATION_COMMON_MESSURE_TIME_H_ */
