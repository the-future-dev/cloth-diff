%% File: `iclr2017.bst'
%% A copy of iclm2010.bst, which is a modification of `plainnl.bst' for use with natbib package 
%%
%% Copyright 2010 Hal Daum\'e III
%% Modified by J. Fürnkranz
%% - Changed labels from (X and Y, 2000) to (X & Y, 2000)
%%
%% Copyright 1993-2007 Patrick W Daly
%% Max-Planck-Institut f\"ur Sonnensystemforschung
%% Max-Planck-Str. 2
%% D-37191 Katlenburg-Lindau
%% Germany
%% E-mail: daly@mps.mpg.de
%%
%% This program can be redistributed and/or modified under the terms
%% of the LaTeX Project Public License Distributed from CTAN
%% archives in directory macros/latex/base/lppl.txt; either
%% version 1 of the License, or any later version.
%%
 % Version and source file information:
 % \ProvidesFile{icml2010.mbs}[2007/11/26 1.93 (PWD)]
 %
 % BibTeX `plainnat' family
 %   version 0.99b for BibTeX versions 0.99a or later,
 %   for LaTeX versions 2.09 and 2e.
 %
 % For use with the `natbib.sty' package; emulates the corresponding
 %   member of the `plain' family, but with author-year citations.
 %
 % With version 6.0 of `natbib.sty', it may also be used for numerical
 %   citations, while retaining the commands \citeauthor, \citefullauthor,
 %   and \citeyear to print the corresponding information.
 %
 % For version 7.0 of `natbib.sty', the KEY field replaces missing
 %   authors/editors, and the date is left blank in \bibitem.
 %
 % Includes field EID for the sequence/citation number of electronic journals
 %  which is used instead of page numbers.
 %
 % Includes fields ISBN and ISSN.
 %
 % Includes field URL for Internet addresses.
 %
 % Includes field DOI for Digital Object Idenfifiers.
 %
 % Works best with the url.sty package of Donald Arseneau.
 %
 % Works with identical authors and year are further sorted by
 %   citation key, to preserve any natural sequence.
 %
ENTRY
  { address
    author
    booktitle
    chapter
    doi
    eid
    edition
    editor
    howpublished
    institution
    isbn
    issn
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    url
    volume
    year
  }
  {}
  { label extra.label sort.label short.list }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}

STRINGS { s t }

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
        { add.period$ write$
          newline$
          "\newblock " write$
        }
        { output.state before.all =
            'write$
            { add.period$ " " * write$ }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

