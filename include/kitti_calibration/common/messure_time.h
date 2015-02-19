/*
 * Messure_time.h
 *
 *  Created on: 19.02.2015
 *      Author: fnolden
 */

#ifndef INCLUDE_KITTI_CALIBRATION_COMMON_MESSURE_TIME_H_
#define INCLUDE_KITTI_CALIBRATION_COMMON_MESSURE_TIME_H_

std::string time_string(clock_t start, clock_t end, std::string name, int iterations, int size1, int size2, int score=0, bool description = false){
	std::stringstream ss;
	std::string spacer = " \t";
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

#endif /* INCLUDE_KITTI_CALIBRATION_COMMON_MESSURE_TIME_H_ */
