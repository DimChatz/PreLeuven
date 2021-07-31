#include<iostream>
#include<cmath>
#include<fstream>
#include<array>
#include<ctime>
#include<cstdlib>


//CODE FOR CREATING INPUT WITH 231 RAYS 30% FROM SOURCE AND 70% FROM ISOTROPIC DATA SET OF AUGER


using namespace std;

int main()
{
// constants
const double pi=3.14159265358979323846264338327950288419716939937;

//initialise streams in&out
ofstream out{"/home/tzikos/Desktop/hope2/30-70-10-new.txt"};

//array for processing input data
array<double,6> a;   //array for in2, see below
array<double,5> c;  //array for in1, see below

//arrays for comparisons/exclusions of duplicates
array<int,69> d;
array<int,162> e;  


//initialize time for rng generator
srand(time(0));


//START OF PROCCESS

//DATA FROM SOURCE
for (int i=0;i<69;i++)
	{	
	int f1=(rand()%230)+1;

rng:f1=(rand()%230)+1;	

	if (i==0) // this is done for first iteration to ensure that the first f1 is not 0
		{
		d[0]=f1;
		}	
	else 
		{
		//comparison-exclusion proccess to ensure that no duplicates are found
		for (int q=0;q<69;q++)
			{
			if (f1==d[q])
				{
				goto rng;
				}
			}
		d[i]=f1;
		}	
	
	//cout<<i<<"\n";
	ifstream in1{"/home/tzikos/Desktop/hope2/CentauriA-10.txt"};
	int count1=1;  //count to find what number is the random generator f1 so that we end up selecting the right UHECR
	for (int j=0;j<230;j++)
		{
		for (int w=0;w<5;w++)
			{
			in1>>c[w];
			}
		
		if (count1==f1)
			{
			out<<c[1]<<" "<<c[2]<<" "<<c[3]<<" "<<c[4]<<"\n";	
			/*if (i==1)
				{
				cout<<c[0]<<" "<<c[1]<<" "<<c[2]<<"\n";    //for checkin//
				} */	 
			count1=1;
			break;
			}	
		else 
			{
			count1++;
			}
		}
	}

//DATA FROM ISOTROPY
for (int i=0;i<162;i++)
	{
	int f2=(rand()%81701)+1;

rng2:f2=(rand()%81701)+1;

	if (i==0)   // this is done for first iteration to ensure that the first f2 is not 0
		{
		e[0]=f2;		
		}
	else
		{
		//comparison-exclusion proccess to ensure that no duplicates are found
		for (int q=0;q<162;q++)
			{
			if (f2==e[q])
				{
				goto rng2;
				}
			}
		e[i]=f2;
		}
	
	ifstream in2{"/home/tzikos/Desktop/hope2/final-list-high-E.txt"};
	int count2=1;	//count to find what number is the random generator f2 so that we end up selecting the right UHECR
	for (int j=0;j<81701;j++)
		{
		for (int w=0;w<6;w++)
			{
			in2>>a[w];
			}
		if (count2==f2)
			{
			out<<a[2]<<" "<<a[3]<<" "<<a[4]<<" "<<a[5]<<"\n";		
			/*if (i==68)
				{
				cout<<count2<<" "<<a[2]<<" "<<a[3]<<"\n";    //for checkin//
				}	*/ 	
			count2=1;
			break;
			}				
		else 
			{
			count2++;
			}
		}
	}


}
