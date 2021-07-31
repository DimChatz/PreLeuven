#include<iostream>
#include<array>
#include<fstream>

using namespace std;

//CODE FOR CREATING A TEXT FILE WITH THE POINTS OF THE CenA SOURCE AS THEY ARRIVED AT EARTH


int main()
{
//arrays for transfering data
array<double,3> a;
array<double,5> b;
char q;

//input-output streams
ifstream in1{"/home/tzikos/Desktop/new/CentauriA-after.txt"};
ofstream out{"/home/tzikos/Desktop/new/CentauriA-after-skymap.txt"};

//start of proccess
for(int i=0;i<297;i++)
	{
	for (int w=0;w<3;w++)
		{
		in1>>a[w];
		}
	ifstream in2{"/home/tzikos/Desktop/new/augerIn.txt"};
	for (int w=1;w<81702;w++)
		{
		in2>>q;
		for (int y=0;y<5;y++)
			{
			in2>>b[y];
			}
		if (w==a[0])
			{
			out<<a[0]<<" "<<b[1]<<" "<<b[2]<<"\n";
			break;			
			}
		}
	}

}
