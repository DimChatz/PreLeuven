#include<iostream>
#include<cmath>
#include<fstream>
#include<array>

//CODE FOR FINDING THE CR's THAT ORIGINATED FROM Cen A

using namespace std;

int main()
{
// constants
const double pi=3.14159265358979323846264338327950288419716939937;

//Centauri A location
double ra0=(13.+25./60+27./3600);
double dec0=(-43.+1./60+9./3600);

//initialise streams in&out
ofstream out{"/home/tzikos/Desktop/hope/CentauriA-10.txt"};
ifstream in{"/home/tzikos/Desktop/hope/final-list-high-E.txt"};


//array for processing input data
array<double,6> a;

//statement of variables
double RA,DEC;

//start of process
for (int i=1;i<=81701;i++)
	{
	for (int j=0;j<6;j++)
		{
		in>>a[j];
		}
	
	DEC=a[5];
	RA=a[4];
	
	if ( (RA<=(ra0+5))&&(RA>=(ra0-5))&&(DEC>=(dec0-5))&&(DEC<=(dec0+5)) )
		{
		out<<a[0]<<" "<<a[2]<<" "<<a[3]<<" "<<a[4]<<" "<<a[5]<<"\n"; 
		}
	}
}
