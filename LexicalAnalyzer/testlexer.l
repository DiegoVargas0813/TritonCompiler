%option noyywrap

%{
    // This section is for C code that will be included in the generated lexer.
    #include <stdio.h>
%}

letter  [a-zA-Z]
digit  [0-9]

%%
[\n\t ]    ; // ignore whitespace
-?(([0-9]+)|([0-9]*\.[0-9]+))([eE][+-]?[0-9]+)?   { printf("NUMBER: %s\n", yytext); } // match integers and floating-point numbers

%%
// This section is for additional C code, such as the main function.
int main(){
    yylex();
}