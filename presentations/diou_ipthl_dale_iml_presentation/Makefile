SHELL=/bin/bash

TEXFILES=$(wildcard presentation*.tex)
PDFFILES=$(patsubst %.tex,%.pdf,$(wildcard presentation*.tex))

all: $(PDFFILES)

%.pdf: %.tex preamble.tex
	xelatex -interaction=nonstopmode -shell-escape $*
	xelatex -interaction=nonstopmode -shell-escape $*
#	biber $*
	rm -f $*.{bak,dvi,aux,log,bbl,blg,ind,ilg,idx,lof,lot,toc,ispell,nav,out,snm,vrb,pyg}
	rm -rf _minted-$*

clean::
	rm -f *~ *.{bak,dvi,aux,log,bbl,blg,ind,ilg,idx,lof,lot,toc,ispell,nav,out,snm,vrb,pyg}
	rm -f $(PDFFILES)
	rm -rf _minted-*
