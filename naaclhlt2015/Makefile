# latex source file name
TARGET=naaclhlt2015

SRC=$(wildcard *.tex) 

PDFCOMMAND=/usr/bin/evince

# If you have have a bibtex file then you need to run this set of commands 
# 1. run latex to generate the aux file
# 2. run bibtex to generate the bbl file from the aux file
# 3. run latex to add the bibliography entries (and generate their
# 	labels)
# 4. run latex again so that any reference to them are updated. 

all:	${TARGET}.tex
	pdflatex ${TARGET}.tex
	bibtex ${TARGET}
	pdflatex ${TARGET}.tex
	pdflatex ${TARGET}.tex
	pdflatex ${TARGET}.tex


view: ${TARGET}.pdf
	${PDFCOMMAND} ${TARGET}.pdf

clean:
	rm -f ${TARGET}.dvi ${TARGET}.log ${TARGET}.pdf ${TARGET}.aux ${TARGET}.bbl  ${TARGET}.blg ${TARGET}.fdb_latexmk ${TARGET}.synctex.gz 

cleanall:
	rm -f ${TARGET}.dvi ${TARGET}.log ${TARGET}.aux ${TARGET}.bbl ${TARGET}.blg ${TARGET}.pdf ${TARGET}.fdb_latexmk ${TARGET}.synctex.gz 

