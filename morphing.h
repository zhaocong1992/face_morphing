#ifndef MOPRPHING_H
#define MOPRPHING_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#include <cv.h>
#include <highgui.h>

#define _CRT_SECURE_NO_WARNINGS


struct Line
{
	CvPoint2D32f P; //start 
	CvPoint2D32f Q; //end
	CvPoint2D32f M; //mid
	double len;
	float degree;
	
	void PQtoMLD();//已知PQ點 
	void MLDtoPQ();//已知中點,長度,角度 
	void show();
	
	double Getu(CvPoint2D32f X);
	double Getv(CvPoint2D32f X);
	CvPoint2D32f Get_Point(double u , double v);
	double Get_Weight(CvPoint2D32f X);
};

struct LinePair
{
	Line leftLine;
	Line rightLine;
	std::vector<struct Line> warpLine;
	
	void genWarpLine();
	void showWarpLine();
};

//======================================

void show_pairs();
void runWarp();

#endif



