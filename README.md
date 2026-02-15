
# How to install. 
You need to have python and FFMPEG installed Correctly. 
The way I installed FFMPEG Is by using the windows Choco Package Manager
https://chocolatey.org/install

Then when running Powershell as Admin type
>choco install ffmpeg-full

Finally go to C:\ProgramData\Blackmagic Design\DaVinci Resolve\Fusion\Scripts\Utility and drop the python file over their. 

Now you should see a script in Workspace>Scripts called Automake. 

If this process takes too long for the legnth of your video just increase the amount of parrel workers
