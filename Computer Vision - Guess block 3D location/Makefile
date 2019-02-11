LIBS = -lopencv_core -lopencv_highgui -lopencv_ccalib -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs -lopencv_features2d -lm

all:	edges hough contours bounding segment1 harris shi-tomasi find-colors pyramid pyr-blobs pyr-edges pyr-blobs-color lsd-lines

edges:	edges.cpp
	g++ -std=c++11 edges.cpp -o edges $(LIBS)

hough:	hough.cpp
	g++ -std=c++11 hough.cpp -o hough $(LIBS)

contours:	contours.cpp
	g++ -std=c++11 contours.cpp -o contours $(LIBS)

bounding:	bounding.cpp
	g++ -std=c++11 bounding.cpp -o bounding $(LIBS)

segment1:	segment1.cpp
	g++ -std=c++11 segment1.cpp -o segment1 $(LIBS)

harris:	harris.cpp
	g++ -std=c++11 harris.cpp -o harris $(LIBS)

shi-tomasi:	shi-tomasi.cpp
	g++ -std=c++11 shi-tomasi.cpp -o shi-tomasi $(LIBS)

find-colors:	find-colors.cpp
	g++ -std=c++11 find-colors.cpp -o find-colors $(LIBS)

pyramid:	pyramid.cpp
	g++ -std=c++11 pyramid.cpp -o pyramid $(LIBS)

pyr-blobs:	pyr-blobs.cpp
	g++ -std=c++11 pyr-blobs.cpp -o pyr-blobs $(LIBS)

pyr-edges:	pyr-edges.cpp
	g++ -std=c++11 pyr-edges.cpp -o pyr-edges $(LIBS)

pyr-blobs-color:	pyr-blobs-color.cpp
	g++ -std=c++11 pyr-blobs-color.cpp -o pyr-blobs-color $(LIBS)

lsd-lines:	lsd-lines.cpp
	g++ -std=c++11 lsd-lines.cpp -o lsd-lines $(LIBS)

clean:
	rm -f edges hough contours bounding segment1 harris shi-tomasi find-colors pyramid pyr-blobs pyr-edges pyr-blobs-color lsd-lines
