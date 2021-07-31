#include<iostream>
#include<array>
#include<fstream>

//CODE FOR CREATING THE SAME INPUTS BUT FOR DIFFERENT Z

using namespace std;

int main()
{
//streams
ifstream in1{"/home/tzikos/Desktop/attempt1-skymap/30-70-new.txt"};
ofstream out2{"/home/tzikos/Desktop/attempt1-skymap/In-Z=2.txt"};
ofstream out6{"/home/tzikos/Desktop/attempt1-skymap/In-Z=6.txt"};
ofstream out26{"/home/tzikos/Desktop/attempt1-skymap/In-Z=26.txt"};

//arrays
array<double,5> a;
array<double,6> b;

//proccess
for (int j=0;j<100;j++)
	{
	for (int i=0;i<5;i++)
		{
		in1>>a[i];
		}
	ifstream in2{"/home/tzikos/Desktop/attempt1-skymap/final-list-high-E.txt"};
	for (int w=0;w<81701;w++)
		{
		for (int q=0;q<6;q++)
			{
			in2>>b[q];
			}
		if (b[0]==a[0])
			{
			out2<<b[1]/2<<" "<<a[1]<<" "<<a[2]<<" "<<"1"<<" "<<"1"<<"\n";
			out6<<b[1]/6<<" "<<a[1]<<" "<<a[2]<<" "<<"1"<<" "<<"1"<<"\n";
			out26<<b[1]/26<<" "<<a[1]<<" "<<a[2]<<" "<<"1"<<" "<<"1"<<"\n";
			in2.close();
			break;
			}
		}		
	}
}
