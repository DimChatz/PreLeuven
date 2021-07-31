#include<iostream>
#include<fstream>
#include<array>
#include<cmath>

// CODE FOR CREATING A LIST WITH ALLA THE INFO ABOUT THE HIGH ENERGY PROPAGATION

using namespace std;

int main()
{

// constants
const double pi=3.14159265358979323846264338327950288419716939937;
const double d0=27.128310056550401*pi/180;
const double a0=(192.8595212503885+90.)*pi/180;
const double L0=(122.93193411101866-90.)*pi/180;

//arrays and variable for stream input,output
array<double,20> b;
array<double,5> a;
char A;
double Bin,Bout,Lin,Lout;
double DECin,DECout,RAin,RAout;

//streams for reading and writing data
ifstream in1{"/home/tzikos/Desktop/new/augerIn.txt"};
ifstream in2{"/home/tzikos/Desktop/new/out.txt"};
ofstream out{"/home/tzikos/Desktop/hope/final-list-high-E.txt"};

for (int i=1; i<=81701;i++)
	{
	in1>>A;
	for  (int j=0;j<5;j++)
		{
		in1>>a[j];
		}	
	for (int j=0;j<20;j++)
		{
		in2>>b[j];
		}
	Bin=a[2]*pi/180;
	Bout=b[3]*pi/180;
	Lin=a[1]*pi/180;
	Lout=b[2]*pi/180;

	DECin=asin(cos(Bin)*cos(d0)*sin(Lin-L0)+sin(Bin)*sin(d0));
	double x=cos(Bin)*cos(Lin-L0);
	double y=(-sin(Bin)*cos(d0)+cos(Bin)*sin(d0)*sin(Lin-L0));
	RAin=atan2(y,x)+a0;
	if (RAin>pi)
	{
	RAin-=pi*2;
	}	
	
	DECout=asin(cos(Bout)*cos(d0)*sin(Lout-L0)+sin(Bout)*sin(d0));
	x=cos(Bout)*cos(Lout-L0);
	y=(-sin(Bout)*cos(d0)+cos(Bout)*sin(d0)*sin(Lout-L0));
	RAout=atan2(y,x)+a0;
	if (RAout>pi)
	{
	RAout-=pi*2;
	}	
	
	out<<i<<" "<<a[0]<<" "<<RAin*180/pi<<" "<<DECin*180/pi<<" "<<RAout*180/pi<<" "<<DECout*180/pi<<"\n";
	}
out<<"#label #Energy #BD-ra #BD-dec #AD-ra #AD-dec";
}
