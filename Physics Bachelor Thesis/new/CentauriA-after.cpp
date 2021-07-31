#include<iostream>
#include<cmath>
#include<fstream>
#include<array>


using namespace std;

int main()
{
// constants
const double pi=3.14159265358979323846264338327950288419716939937;
const double d0=27.128310056550401*pi/180;
const double a0=(192.8595212503885+90.)*pi/180;
const double L0=(122.93193411101866-90.)*pi/180;


//Centauri A location
double ra0=(13.+25./60+27./3600)*15*pi/180;
double dec0=(-43.+1./60+9./3600)*pi/180;
double lat0,long0;
lat0=asin(-cos(dec0)*cos(d0)*sin(ra0-a0)+sin(dec0)*sin(d0));
double y=(sin(dec0)*cos(d0)+cos(dec0)*sin(d0)*sin(ra0-a0));
double x=cos(dec0)*cos(ra0-a0);
long0=(atan2(y,x)+L0);
if (long0>pi)
	{
	long0-=pi*2;
	}
cout<<"\n"<<long0*180/pi<<"\t"<<lat0*180/pi<<"\n";


//initialise streams in&out
ofstream out{"/home/tzikos/Desktop/new/CentauriA-after.txt"};
ifstream in{"/home/tzikos/Desktop/new/out.txt"};


//array for processing input data
array<double,20> a;

//statement of variables
double LONG,LAT;



//start of process
for (int i=1;i<=81701;i++)
	{
	for (int j=0;j<20;j++)
		{
		in>>a[j];
		}
	
	LAT=a[3];
	LONG=a[2];
	
	//cout<<LONG<<"\t"<<LAT<<"\n";;
	


	if ( (LONG<=-45.4804)&&(LONG>=-55.4804)&&(LAT>=14.4554)&&(LAT<=24.4554) )
		{
		out<<i<<" "<<LONG<<" "<<LAT<<"\n"; 
		}
	}
}
