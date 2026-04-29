%{
    // Lexical Analyzer for the Python-based programming language Triton.
    // The lexer is designed to recognize common keywords, identifies, numbers, comments, strings and operators.
    // It accepts a file as input and produces a token list with IDs and value in a symbol table in the case of numbers, identifiers and strings.
    // Like the previous line suggests, it also outputs a symbol table for identifiers, numbers and strings, which is used by the parser to check for semantic errors and to generate code.

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    // Enum for token types
    enum TokenIds {
        TOKEN_DEF = 1,
        TOKEN_IF = 2,
        TOKEN_ELSE = 3,
        TOKEN_FOR = 4,
        TOKEN_WHILE = 5,
        TOKEN_HASH = 6,
        TOKEN_DOT = 7,
        TOKEN_COMMENT = 8,
        TOKEN_AT = 9,
        TOKEN_PLUS = 10,
        TOKEN_MINUS = 11,
        TOKEN_MULTIPLY = 12,
        TOKEN_DIVIDE = 13,
        TOKEN_MODULO = 14,
        TOKEN_GREATER = 15,
        TOKEN_LESS = 16,
        TOKEN_EQUALS = 17,
        TOKEN_COLON = 18,
        TOKEN_COMMA = 19,
        TOKEN_OPAREN = 20,
        TOKEN_CPAREN = 21,
        TOKEN_OBRACKET = 22,
        TOKEN_CBRACKET = 23,
        TOKEN_NUMBER = 24,
        TOKEN_IDENTIFIER = 25,
        TOKEN_STRING = 26,
    }

    // Counter variables for token IDs and symbol table entries.
    static int id_count = 0; // Counter for identifiers.
    static int num_count = 0; // Counter for numbers.
    static int str_count = 0; // Counter for strings.
    static int token_count = 0; // Counter for total tokens.

    // Definition of the token list and symbol table.
    #define MAX_TOKENS 4096

    static identifiers symbol_tab


    static pair<int, char*> token_list[MAX_TOKENS]; // List of tokens with their IDs and values (if applicable).
        

        return 0; // Default case for unknown tokens.
    }

    // NOTE: char* in C is a string literal. 
    static void add_token(const char* token_name, int entry){
        int token_id;

        token_id = token_id_from_name(token_name);
        if(token_id == 0){
            fprintf(stderr, "Unknown token: %s\n", token_name);
            exit(1);
        }

        token_list[token_count]
    }


%}

LETTER    [a-zA-Z]
ID        ({LETTER}|_)( {LETTER}|{DIGIT}|_)* 
DIGIT     [0-9]
INT      {DIGIT}+
FLOAT    {DIGIT}+"."{DIGIT}+
NUMBER   -?({FLOAT}|{INT})

/* Strings basicaly accept any amount of characters excluding new line, non escaped backslashes or quotations. The only way to allow backslashes or quotations is followed by a backslash.  */
STRING   \"([^\\\n\"]|\\.)*\"
%%
[ \t\r\n]+           ; // Ignore whitespace

#[^\n]*            ; // Skip python-style comments. Multi-lines are ignored since this is a lexer using minimal rules.


"def"                { printf("KEYWORD: def\n"); } // Keywords
"if"                 { printf("KEYWORD: if\n"); }
"else"               { printf("KEYWORD: else\n"); }
"for"                { printf("KEYWORD: for\n"); }
"while"              { printf("KEYWORD: while\n"); }

"#"                { /* Skip comments */ } // TODO: Add logic to detect token and push to token list and symbol table if needed.
"."
"@"
"+"
"-"
"*"
"/"
"%"
">"
"<"
"="
":"
"("
")"
"["
"]"

{STRING}


{NMUMBER}

{ID}

.                    {
    printf("UNKNOWN TOKEN: %s\n", yytext);
}
%%
