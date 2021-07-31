#include<iostream>
#include<cmath>
#include<fstream>
#include<array>
#include<ctime>
#include<cstdlib>

//CODE FOR CREATING THE INPUT THE CRT CODE ACCEPTS 

using namespace std;

int main()
{
// constants
const double pi=3.14159265358979323846264338327950288419716939937;
//constants for transforming between eq and ga
const double d0=27.128310056550401*pi/180;
const double a0=(192.8595212503885+90.)*pi/180;
const double L0=(122.93193411101866-90.)*pi/180;

//initialise streams in&out
ifstream in1{"/home/tzikos/Desktop/events_4-8.dat"};   // events of CR's
ifstream in2{"/home/tzikos/Desktop/new/Energies.dat"}; //contains high energy lists
ofstream out{"/home/tzikos/Desktop/new/augerIn.txt"};

//array for processing input data
array<double,6> a;
array<double,231> b;

//statement of variables
double RA,DEC,LONG,LAT;

//array for energies
for (int w=0;w<=230;w++)
		{
		//take in the energy data
		in2>>b[w];
		}


//initialize time for rng generator
srand(time(0));


//start of process
for (int i=0;i<81701;i++)
	{
	for (int j=0;j<6;j++)
		{
		in1>>a[j];
		}
	
	DEC=a[2]*pi/180;
	RA=a[3]*pi/180;
	
	//code for ga to eq
	LAT=asin(-cos(DEC)*cos(d0)*sin(RA-a0)+sin(DEC)*sin(d0));
	double y=(sin(DEC)*cos(d0)+cos(DEC)*sin(d0)*sin(RA-a0));
	double x=cos(DEC)*cos(RA-a0);
	LONG=(atan2(y,x)+L0);
	if (LONG>pi)
        	{
		LONG-=pi*2;
		}
	LONG*=180/pi;
	LAT*=180/pi;

	//rng generator for energies
	int r=(rand()%231);	
	
	//output to file
	out<<"C "<<b[r]<<" "<<LONG<<" "<<LAT<<" "<<"1 1\n"; // form of CRT input for cosmic rays
	}
}
