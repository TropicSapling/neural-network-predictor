@echo off

setlocal enabledelayedexpansion

set I=0

for /F "tokens=*" %%k in (%1) do (
	set /A I=!I! + 1
	set LINE!I!=%%k
)

break>"%1"
for /L %%c in (!I!,-1,1) do (
	echo !LINE%%c!>> "%1"
)
