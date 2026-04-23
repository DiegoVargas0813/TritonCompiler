%{
    // Lexical analyzer for the Triton language.
    // Input: Triton source code. Output: Symbol tables for identifiers, numbers, srings, and a token list.
    // This file defines the tokens and patterns for the Triton language.
    // Analyzes code and returns symbol tables for identifiers, numbers and strings.
    // Each table has an entry number and lexeme column, the number table also having a type column.
    // The lexer will also return a token list that whose entries can be a pair of the identifier of the token and the entry number of the symbol table or just the identifier in the case of operators, keywords and delimiters.
    // EX:
    // Input: int x = 5;
    // Output:
    // Symbol Table for Identifiers:
    // Entry Number | Lexeme
    // 1            | x
    // Symbol Table for Numbers:
    // Entry Number | Lexeme | Type
    // 1            | 5      | int
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    #define MAX_ENTRIES 4096
    #define MAX_LEXEME_LEN 256

    typedef struct {
        int entry;
        char lexeme[MAX_LEXEME_LEN];
    } IdentifierEntry;

    typedef struct {
        int entry;
        char lexeme[MAX_LEXEME_LEN];
        char type[16];
    } NumberEntry;

    typedef struct {
        int entry;
        char lexeme[MAX_LEXEME_LEN];
    } StringEntry;

    static IdentifierEntry identifiers[MAX_ENTRIES];
    static NumberEntry numbers[MAX_ENTRIES];
    static StringEntry strings[MAX_ENTRIES];

    static int id_count = 0;
    static int num_count = 0;
    static int str_count = 0;

    static char token_list[MAX_ENTRIES * 4][128];
    static int token_count = 0;

    static int add_identifier(const char* lexeme) {
        int i;
        for (i = 0; i < id_count; i++) {
            if (strcmp(identifiers[i].lexeme, lexeme) == 0) {
                return identifiers[i].entry;
            }
        }

        if (id_count >= MAX_ENTRIES) {
            fprintf(stderr, "Identifier table overflow.\n");
            exit(1);
        }

        identifiers[id_count].entry = id_count + 1;
        strncpy(identifiers[id_count].lexeme, lexeme, MAX_LEXEME_LEN - 1);
        identifiers[id_count].lexeme[MAX_LEXEME_LEN - 1] = '\0';
        id_count++;
        return id_count;
    }

    static int add_number(const char* lexeme) {
        int i;
        int is_float = (strchr(lexeme, '.') != NULL) || (strchr(lexeme, 'e') != NULL) || (strchr(lexeme, 'E') != NULL);

        for (i = 0; i < num_count; i++) {
            if (strcmp(numbers[i].lexeme, lexeme) == 0) {
                return numbers[i].entry;
            }
        }

        if (num_count >= MAX_ENTRIES) {
            fprintf(stderr, "Number table overflow.\n");
            exit(1);
        }

        numbers[num_count].entry = num_count + 1;
        strncpy(numbers[num_count].lexeme, lexeme, MAX_LEXEME_LEN - 1);
        numbers[num_count].lexeme[MAX_LEXEME_LEN - 1] = '\0';
        strcpy(numbers[num_count].type, is_float ? "float" : "int");
        num_count++;
        return num_count;
    }

    static int add_string(const char* lexeme) {
        int i;
        for (i = 0; i < str_count; i++) {
            if (strcmp(strings[i].lexeme, lexeme) == 0) {
                return strings[i].entry;
            }
        }

        if (str_count >= MAX_ENTRIES) {
            fprintf(stderr, "String table overflow.\n");
            exit(1);
        }

        strings[str_count].entry = str_count + 1;
        strncpy(strings[str_count].lexeme, lexeme, MAX_LEXEME_LEN - 1);
        strings[str_count].lexeme[MAX_LEXEME_LEN - 1] = '\0';
        str_count++;
        return str_count;
    }

    static void add_token(const char* token_name, int entry) {
        if (token_count >= MAX_ENTRIES * 4) {
            fprintf(stderr, "Token list overflow.\n");
            exit(1);
        }

        if (entry > 0) {
            snprintf(token_list[token_count], sizeof(token_list[token_count]), "%s(%d)", token_name, entry);
        } else {
            snprintf(token_list[token_count], sizeof(token_list[token_count]), "%s", token_name);
        }
        token_count++;
    }

    static void print_results(void) {
        int i;

        printf("\nSymbol Table for Identifiers:\n");
        printf("Entry Number | Lexeme\n");
        for (i = 0; i < id_count; i++) {
            printf("%-12d | %s\n", identifiers[i].entry, identifiers[i].lexeme);
        }

        printf("\nSymbol Table for Numbers:\n");
        printf("Entry Number | Lexeme | Type\n");
        for (i = 0; i < num_count; i++) {
            printf("%-12d | %-6s | %s\n", numbers[i].entry, numbers[i].lexeme, numbers[i].type);
        }

        printf("\nSymbol Table for Strings:\n");
        printf("Entry Number | Lexeme\n");
        for (i = 0; i < str_count; i++) {
            printf("%-12d | %s\n", strings[i].entry, strings[i].lexeme);
        }

        printf("\nToken List:\n");
        for (i = 0; i < token_count; i++) {
            printf("%s\n", token_list[i]);
        }
    }
%}

%option noyywrap

LETTER      [a-zA-Z_]
DIGIT       [0-9]
ID          {LETTER}({LETTER}|{DIGIT})*
INT         {DIGIT}+
FLOAT       ({DIGIT}+\.{DIGIT}+|{DIGIT}+\.|\.{DIGIT}+)([eE][+-]?{DIGIT}+)?
SCI         {DIGIT}+([eE][+-]?{DIGIT}+)
NUMBER      {FLOAT}|{SCI}|{INT}
STRING      \"([^\\\n\"]|\\.)*\"

%%

[ \t\r\n]+                                      ;

"//"[^\n]*                                      { add_token("COMMENT", 0); }
"/*"([^*]|\*+[^*/])*\*+"/"                    { add_token("COMMENT", 0); }

"def"                                            { add_token("DEF", 0); }
"if"                                             { add_token("IF", 0); }
"else"                                           { add_token("ELSE", 0); }
"for"                                            { add_token("FOR", 0); }
"while"                                          { add_token("WHILE", 0); }

"#"                                              { add_token("HASH_TAG", 0); }
"."                                              { add_token("DOT", 0); }
"@"                                              { add_token("AT_SIGN", 0); }
"+"                                              { add_token("PLUS_SIGN", 0); }
"-"                                              { add_token("MINUS_SIGN", 0); }
"*"                                              { add_token("MULTIPLICATION_SIGN", 0); }
"/"                                              { add_token("DIVISION_SIGN", 0); }
"%"                                              { add_token("MODULO", 0); }
">"                                              { add_token("GREATER_THAN", 0); }
"<"                                              { add_token("LESS_THAN", 0); }
"="                                              { add_token("EQUAL", 0); }
":"                                              { add_token("SEMICOLON", 0); }
","                                              { add_token("COMMA", 0); }
"("                                              { add_token("OPEN_PARENTHESIS", 0); }
")"                                              { add_token("CLOSE_PARENTHESIS", 0); }
"["                                              { add_token("OPEN_SQUARE_BRACKET", 0); }
"]"                                              { add_token("CLOSE_SQUARE_BRACKET", 0); }

{STRING}                                          {
                                                    int entry = add_string(yytext);
                                                    add_token("STRING", entry);
                                                  }

{NUMBER}                                          {
                                                    int entry = add_number(yytext);
                                                    add_token("NUMBER", entry);
                                                  }

{ID}                                              {
                                                    int entry = add_identifier(yytext);
                                                    add_token("IDENTIFIER", entry);
                                                  }

.                                                 {
                                                    fprintf(stderr, "Unknown token: %s\n", yytext);
                                                  }

%%

int main(int argc, char* argv[]) {
    const char* input_path;
    const char* output_path;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <input_file> [output_txt]\n", argv[0]);
        return 1;
    }

    input_path = argv[1];
    output_path = (argc == 3) ? argv[2] : "lexer_output.txt";

    if (freopen(input_path, "r", stdin) == NULL) {
        fprintf(stderr, "Could not open input file: %s\n", input_path);
        return 1;
    }

    if (freopen(output_path, "w", stdout) == NULL) {
        fprintf(stderr, "Could not open output file: %s\n", output_path);
        return 1;
    }

    yylex();
    print_results();

    fprintf(stderr, "Lexer output written to: %s\n", output_path);
    return 0;
}