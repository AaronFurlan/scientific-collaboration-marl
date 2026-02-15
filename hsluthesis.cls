%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hsluthesis.cls
%
% (c) Lucerne University of Applied Sciences and Arts
%
% The official LaTeX template class for HSLU theses. It ensures the thesis style
% complies with HSLU requirements for various study programs at HSLU-I.
%
% This class has been created by Ramón Christen at the Lucerne University of Applied Sciences and Arts.
%
% Created for HSLU by Ramón Christen 2025
%
%   (c) Lucerne University of Applied Sciences and Arts
%   v1.0    22-09-2025:     Original version
%   v1.1    29-09-2025:     Extended with WIPRO option
%   v1.2    10-10-2025:     Minor corrections for WIPRO
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Identification
%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}

% name and release date
\ProvidesClass{hsluthesis}[2025/06/01 Official HSLU Thesis Template]



% PDF/A Document compliance
%%%%%%%%%%%%%%%%%%
\pdfcatalog{
  /StructTreeRoot <<                                     % Define the structure tree root for document tagging
    /Type /StructTreeRoot                                % Specify that this is a structure tree root
    /K []                                                % Placeholder for structure elements (empty for now)
  >>                            
  /MarkInfo << /Marked true >>                           % Ensure the document is marked as tagged for accessibility
}


% Declare options
%%%%%%%%%%%%%%%%%%
% template conditions set with options
\newif\ifnoip                                           % intellectual property section included
\newif\ifacceptsheet                                    % accept sheet included
\newif\ifcedu                                           % continuous education (CAS, SAS)
\newif\ifgroupwork                                      % extends declaration for two students
\newif\ifnograd                                         % no graduation (e.g. WIPRO)
\newif\ifshortpresdec                                   % short presentation declaration (after report type on title page)

% options treatment
\DeclareOption{bachelor}{}
\DeclareOption{master}{}
\DeclareOption{cas}{%
    \cedutrue
    \DeclareOption{noip}{\noiptrue}
}
\DeclareOption{sas}{
    \cedutrue
    \shortpresdectrue
    \DeclareOption{noip}{\noiptrue}
}
\DeclareOption{wipro}{
    \nogradtrue
    \shortpresdectrue
}
\DeclareOption{groupwork}{
    \groupworktrue
}
\DeclareOption{acceptancesheet}{\acceptsheettrue}

% pass additional options to parent class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}

\ProcessOptions\relax


% Class and package load
%%%%%%%%%%%%%%%%
\LoadClass[12pt, a4paper, oneside]{book}                 % document style definition

\RequirePackage{babel}
\RequirePackage{graphics}
\RequirePackage{rotating}
\RequirePackage{epstopdf}
\RequirePackage[export]{adjustbox}
\RequirePackage[utf8]{inputenc}
\RequirePackage{hsluthesisterms}                         % apply HSLU style

\RequirePackage{enumitem,amssymb}
\newlist{todolist}{itemize}{2}
\setlist[todolist]{label=$\square$}


% commands
%%%%%%%%%%%%%%%%
% general commands
\newcommand*{\fullstop}[1]{#1.}


% Package Body
%%%%%%%%%%%%%%%%
% set up the page dimensions. This assumes a4 paper.
% vertical
\voffset -15mm
\topmargin 0mm
\headheight 5mm
\headsep 12.5mm
\textheight 235mm

% horizontal
\oddsidemargin 10mm
\evensidemargin 0mm
\textwidth 147mm

\reversemarginpar

\DeclareFontFamily{\encodingdefault}{\ttdefault}{\hyphenchar\font=`\-} % Allow hyphen in texttt

% Create abstract environment
\newlength\tindent
\newenvironment{abstractstyle}[1]{%
    \cleardoublepage
    \begin{center}\LARGE\textbf{#1}\end{center}
    \vspace{1cm}
    
    \setlength{\tindent}{\parindent}
    \setlength{\parindent}{0pt}
    }{%
    \vspace{1cm}
    \setlength{\parindent}{\tindent}
}

% title page
\renewcommand*{\maketitle}{%
    \begin{titlepage}%
        \begin{center}%
            \begin{figure}
                \centering
                    \includegraphics[height=.07\textwidth,valign=t]{\logo}
                \hfill
                \vspace{20mm}
            \end{figure}
            
            {\large \depname \par \uniname{} (\unicountry)}
            
            \vskip 20mm%
            {\LARGE \textsc \@title \\ \subtit \vskip \baselineskip \par}%
        
            \vskip 20mm
            {\LARGE \textsc \thesistypetxt \par}
            \vskip 3mm
            {\small \fullstop{\ifshortpresdec\presentedto\else\presentedtoacgrade\fi}}
        
            \vskip 20mm
            {\by \par}
            \vskip 5mm
            {\LARGE \textbf{\@author} \par}%
            \vskip 5mm
            {\from \par}
            \vskip 5mm
            {\LARGE \orig}
        	
            \vfill%
            
            % {\textsc Thesis No.: \thesisnum
            % \par Uniprint
            % \par \@date}
            
        \end{center}
    \end{titlepage}%
    
    % set roman page numbering for preamble
    \pagenumbering{gobble}

    % page: acceptance sheet
    \ifacceptsheet
        \noindent \accept@txt
        \jurymembers
    
    
        \noindent \defencelocation, \defensedatevar
        
        \begin{table}[b]
            \centering
            \begin{tabular}{p{.5\textwidth} p{.5\textwidth}}
                 \textbf{\supervis}\newline \sign@supervisor{} & \textbf{\deanoffaculty}\newline \sign@facdean{} \\ & \\ & \\ & \\ & \\ & \\ & \\ & \\ & 
            \end{tabular}
        \end{table}
    \fi

    \frontmatter
    \clearpage
    
    % page: declaration
    \begin{abstractstyle}{\cons@hmain}
        {\Large \noindent \cons@hi}
        
        \vskip 5mm
        \begin{tabular}{l l}
            \cons@title:  & \@title \\
            \\[-4mm]
            \cons@student:  & \@author \\
            \\[-4mm]
            \ifgroupwork
                \cons@student:  & \@author \\
                \\[-4mm]
            \fi
            \cons@studytopic: & \ifcedu\academicgrade\else\academictitle\fi\\
            \\[-4mm]
            \ifnograd\cons@year\else\cons@defyear\fi: & \@date \\
            \\[-4mm]
            \cons@supervisor: & \supervisorvar \\
            \\[-4mm]
            \cons@expert: & \extexp \\
            \\[-4mm]
            \cons@indpartner: & \indpart \\
        \end{tabular}
     
        \vskip 12mm
        {\Large \noindent \cons@hii}
    
        \vskip 5mm
        \begin{todolist}
            \item \cons@public
            \vspace{-2mm}
            \item \cons@confidential
        \end{todolist}
        \vspace{-4mm}
         
        \vskip 12mm
        {\Large \noindent \cons@hiii }
    
        \vskip 5mm
        \noindent \cons@self
         
        \vskip 10mm
        \noindent \cons@sign{} \underline{\hspace{5cm}}
        \ifgroupwork
            \vskip 7mm
            \noindent \cons@sign{} \underline{\hspace{5cm}}
        \fi

        % page: experssion of thanks and gratitute
        \clearpage
        {\Large \noindent \cons@thanks}
        
        \vskip 5mm
        \acknowledgments

        \vskip 3.5mm
        \raggedleft
        \@author, \@date

        \raggedright
        \vfill
        \ifnoip
        \else
            \noindent\textit{\cons@ip}
        \fi
        
    \end{abstractstyle}

    \setcounter{footnote}{0}%
}