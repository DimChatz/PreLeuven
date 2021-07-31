#include<iostream>
#include<array>
#include<fstream>
#include<cmath>

//CODE FOR FINDING THE NEW SOURCE WE WILL USE

using namespace std;

int main()
{
//pi
const double pi=3.14159265358979323846264338327950288419716939937;
//streams
ifstream in{"/home/tzikos/Desktop/attempt1/final-list-high-E.txt"};
ofstream out{"/home/tzikos/Desktop/attempt1/diff-calc-2pt.txt"};

//variables
double dRA,dDEC;
double x1,y1,z1,x2,y2,z2;
double ra1,ra2,dec1,dec2,pt2;
int count=0;
array<double,6> a;

for (int i=0;i<81701;i++)
	{
	for (int j=0;j<6;j++)
		{	
		in>>a[j];
		}
	//calculating 2pt
	ra1=a[2]*pi/180;
	ra2=a[4]*pi/180;
	dec1=a[3]*pi/180;
	dec2=a[5]*pi/180;	
	x1=sin(pi/2-dec1)*cos(ra1);
	y1=sin(pi/2-dec1)*sin(ra1);
	z1=cos(pi/2-dec1);
	x2=sin(pi/2-dec2)*cos(ra2);
	y2=sin(pi/2-dec2)*sin(ra2);
	z2=cos(pi/2-dec2);
	pt2=acos(x1*x2+y1*y2+z1*z2);
	pt2=pt2*180/pi;
	if (pt2>20)
		{
		out<<a[0]<<" "<<a[2]<<" "<<a[3]<<" "<<a[4]<<" "<<a[5]<<" "<<pt2<<"\n";
		count++;
		}
	}
cout<<"\n"<<count<<"\n";
}
