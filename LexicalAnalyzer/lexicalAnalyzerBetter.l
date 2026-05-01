%{
    // Lexical analyzer for the Triton language.
    // Input: Triton source code. Output: Symbol tables for identifiers, numbers, strings, and a token list.
    // This file defines the tokens and patterns for the Triton language.
    // Analyzes code and returns symbol tables for identifiers, numbers and strings.
    // Each table has an entry number and lexeme column, the number table also having a type column.
    // The lexer will also return a token list whose entries can be a pair of the token id and the
    // symbol table entry number, or just the token id in the case of operators, keywords and delimiters.
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

    typedef struct {
        int token_id;
        int symbol_table_entry;
    } TokenEntry;

    // Token IDs — must stay consistent with the token_ids list printed in print_results()
    enum TokenId {
        TOK_COMMENT              = 1,
        TOK_DEF                  = 2,
        TOK_IF                   = 3,
        TOK_ELSE                 = 4,
        TOK_FOR                  = 5,
        TOK_WHILE                = 6,
        TOK_IMPORT               = 7,
        TOK_AS                   = 8,
        TOK_IN                   = 9,
        TOK_TRUE                 = 10,
        TOK_FALSE                = 11,
        TOK_NONE                 = 12,
        TOK_DOT                  = 13,
        TOK_AT_SIGN              = 14,
        TOK_PLUS_SIGN            = 15,
        TOK_MINUS_SIGN           = 16,
        TOK_MULTIPLICATION_SIGN  = 17,
        TOK_DIVISION_SIGN        = 18,
        TOK_FLOOR_DIVISION       = 19,
        TOK_MODULO               = 20,
        TOK_GREATER_THAN         = 21,
        TOK_LESS_THAN            = 22,
        TOK_EQUAL                = 23,
        TOK_DOUBLE_EQUAL         = 24,
        TOK_NOT_EQUAL            = 25,
        TOK_LESS_EQUAL           = 26,
        TOK_GREATER_EQUAL        = 27,
        TOK_BITWISE_AND          = 28,
        TOK_BITWISE_XOR          = 29,
        TOK_COLON                = 30,
        TOK_COMMA                = 31,
        TOK_OPEN_PARENTHESIS     = 32,
        TOK_CLOSE_PARENTHESIS    = 33,
        TOK_OPEN_SQUARE_BRACKET  = 34,
        TOK_CLOSE_SQUARE_BRACKET = 35,
        TOK_STRING               = 36,
        TOK_NUMBER               = 37,
        TOK_IDENTIFIER           = 38
    };

    static int id_count  = 0;
    static int num_count = 0;
    static int str_count = 0;

    static TokenEntry token_list[MAX_ENTRIES * 4];
    static int token_count = 0;

    // Escapes special characters when printing JSON strings
    static void print_json_string(const char* s) {
        const unsigned char* p = (const unsigned char*)s;
        putchar('"');
        while (*p) {
            switch (*p) {
                case '"':  printf("\\\""); break;
                case '\\': printf("\\\\"); break;
                case '\b': printf("\\b");  break;
                case '\f': printf("\\f");  break;
                case '\n': printf("\\n");  break;
                case '\r': printf("\\r");  break;
                case '\t': printf("\\t");  break;
                default:
                    if (*p < 0x20) printf("\\u%04x", *p);
                    else           putchar(*p);
                    break;
            }
            p++;
        }
        putchar('"');
    }

    // Maps a token name string to its numeric ID
    static int token_id_from_name(const char* name) {
        if (strcmp(name, "COMMENT")              == 0) return TOK_COMMENT;
        if (strcmp(name, "DEF")                  == 0) return TOK_DEF;
        if (strcmp(name, "IF")                   == 0) return TOK_IF;
        if (strcmp(name, "ELSE")                 == 0) return TOK_ELSE;
        if (strcmp(name, "FOR")                  == 0) return TOK_FOR;
        if (strcmp(name, "WHILE")                == 0) return TOK_WHILE;
        if (strcmp(name, "IMPORT")               == 0) return TOK_IMPORT;
        if (strcmp(name, "AS")                   == 0) return TOK_AS;
        if (strcmp(name, "IN")                   == 0) return TOK_IN;
        if (strcmp(name, "TRUE")                 == 0) return TOK_TRUE;
        if (strcmp(name, "FALSE")                == 0) return TOK_FALSE;
        if (strcmp(name, "NONE")                 == 0) return TOK_NONE;
        if (strcmp(name, "DOT")                  == 0) return TOK_DOT;
        if (strcmp(name, "AT_SIGN")              == 0) return TOK_AT_SIGN;
        if (strcmp(name, "PLUS_SIGN")            == 0) return TOK_PLUS_SIGN;
        if (strcmp(name, "MINUS_SIGN")           == 0) return TOK_MINUS_SIGN;
        if (strcmp(name, "MULTIPLICATION_SIGN")  == 0) return TOK_MULTIPLICATION_SIGN;
        if (strcmp(name, "DIVISION_SIGN")        == 0) return TOK_DIVISION_SIGN;
        if (strcmp(name, "FLOOR_DIVISION")       == 0) return TOK_FLOOR_DIVISION;
        if (strcmp(name, "MODULO")               == 0) return TOK_MODULO;
        if (strcmp(name, "GREATER_THAN")         == 0) return TOK_GREATER_THAN;
        if (strcmp(name, "LESS_THAN")            == 0) return TOK_LESS_THAN;
        if (strcmp(name, "EQUAL")                == 0) return TOK_EQUAL;
        if (strcmp(name, "DOUBLE_EQUAL")         == 0) return TOK_DOUBLE_EQUAL;
        if (strcmp(name, "NOT_EQUAL")            == 0) return TOK_NOT_EQUAL;
        if (strcmp(name, "LESS_EQUAL")           == 0) return TOK_LESS_EQUAL;
        if (strcmp(name, "GREATER_EQUAL")        == 0) return TOK_GREATER_EQUAL;
        if (strcmp(name, "BITWISE_AND")          == 0) return TOK_BITWISE_AND;
        if (strcmp(name, "BITWISE_XOR")          == 0) return TOK_BITWISE_XOR;
        if (strcmp(name, "COLON")                == 0) return TOK_COLON;
        if (strcmp(name, "COMMA")                == 0) return TOK_COMMA;
        if (strcmp(name, "OPEN_PARENTHESIS")     == 0) return TOK_OPEN_PARENTHESIS;
        if (strcmp(name, "CLOSE_PARENTHESIS")    == 0) return TOK_CLOSE_PARENTHESIS;
        if (strcmp(name, "OPEN_SQUARE_BRACKET")  == 0) return TOK_OPEN_SQUARE_BRACKET;
        if (strcmp(name, "CLOSE_SQUARE_BRACKET") == 0) return TOK_CLOSE_SQUARE_BRACKET;
        if (strcmp(name, "STRING")               == 0) return TOK_STRING;
        if (strcmp(name, "NUMBER")               == 0) return TOK_NUMBER;
        if (strcmp(name, "IDENTIFIER")           == 0) return TOK_IDENTIFIER;
        return 0;
    }

    // Adds an identifier to the table if not already present; returns its entry number
    static int add_identifier(const char* lexeme) {
        int i;
        for (i = 0; i < id_count; i++) {
            if (strcmp(identifiers[i].lexeme, lexeme) == 0)
                return identifiers[i].entry;
        }
        if (id_count >= MAX_ENTRIES) { fprintf(stderr, "Identifier table overflow.\n"); exit(1); }
        identifiers[id_count].entry = id_count + 1;
        strncpy(identifiers[id_count].lexeme, lexeme, MAX_LEXEME_LEN - 1);
        identifiers[id_count].lexeme[MAX_LEXEME_LEN - 1] = '\0';
        id_count++;
        return id_count;
    }

    // Detects int vs float/scientific and adds to the number table; returns entry number
    static int add_number(const char* lexeme) {
        int i;
        int is_float = (strchr(lexeme, '.') != NULL)
                    || (strchr(lexeme, 'e') != NULL)
                    || (strchr(lexeme, 'E') != NULL);
        for (i = 0; i < num_count; i++) {
            if (strcmp(numbers[i].lexeme, lexeme) == 0)
                return numbers[i].entry;
        }
        if (num_count >= MAX_ENTRIES) { fprintf(stderr, "Number table overflow.\n"); exit(1); }
        numbers[num_count].entry = num_count + 1;
        strncpy(numbers[num_count].lexeme, lexeme, MAX_LEXEME_LEN - 1);
        numbers[num_count].lexeme[MAX_LEXEME_LEN - 1] = '\0';
        strcpy(numbers[num_count].type, is_float ? "float" : "int");
        num_count++;
        return num_count;
    }

    // Strips surrounding quotes (single or double) before storing; returns entry number
    static int add_string(const char* lexeme) {
        int i;
        size_t len = strlen(lexeme);
        const char* start = lexeme;
        size_t copy_len = len;

        // Strip surrounding quotes — works for both ' and "
        if (len >= 2 && (lexeme[0] == '"' || lexeme[0] == '\'')
                     && lexeme[len - 1] == lexeme[0]) {
            start    = lexeme + 1;
            copy_len = len - 2;
        }
        if (copy_len >= MAX_LEXEME_LEN) copy_len = MAX_LEXEME_LEN - 1;

        for (i = 0; i < str_count; i++) {
            if (strncmp(strings[i].lexeme, start, copy_len) == 0
                    && strings[i].lexeme[copy_len] == '\0')
                return strings[i].entry;
        }
        if (str_count >= MAX_ENTRIES) { fprintf(stderr, "String table overflow.\n"); exit(1); }
        strings[str_count].entry = str_count + 1;
        strncpy(strings[str_count].lexeme, start, copy_len);
        strings[str_count].lexeme[copy_len] = '\0';
        str_count++;
        return str_count;
    }

    // Appends one token (with optional symbol-table reference) to the token list
    static void add_token(const char* token_name, int entry) {
        int token_id;
        if (token_count >= MAX_ENTRIES * 4) { fprintf(stderr, "Token list overflow.\n"); exit(1); }
        token_id = token_id_from_name(token_name);
        if (token_id == 0) { fprintf(stderr, "Unknown token name: %s\n", token_name); exit(1); }
        token_list[token_count].token_id          = token_id;
        token_list[token_count].symbol_table_entry = entry;
        token_count++;
    }

    // Prints the full JSON output: token catalogue, symbol tables, and token list
    static void print_results(void) {
        int i;
        printf("{\n");

        // Token ID catalogue — must match the enum above
        printf("  \"token_ids\": [\n");
        printf("    [1,\"COMMENT\"],[2,\"DEF\"],[3,\"IF\"],[4,\"ELSE\"],[5,\"FOR\"],[6,\"WHILE\"],\n");
        printf("    [7,\"IMPORT\"],[8,\"AS\"],[9,\"IN\"],[10,\"TRUE\"],[11,\"FALSE\"],[12,\"NONE\"],\n");
        printf("    [13,\"DOT\"],[14,\"AT_SIGN\"],[15,\"PLUS_SIGN\"],[16,\"MINUS_SIGN\"],\n");
        printf("    [17,\"MULTIPLICATION_SIGN\"],[18,\"DIVISION_SIGN\"],[19,\"FLOOR_DIVISION\"],[20,\"MODULO\"],\n");
        printf("    [21,\"GREATER_THAN\"],[22,\"LESS_THAN\"],[23,\"EQUAL\"],[24,\"DOUBLE_EQUAL\"],\n");
        printf("    [25,\"NOT_EQUAL\"],[26,\"LESS_EQUAL\"],[27,\"GREATER_EQUAL\"],\n");
        printf("    [28,\"BITWISE_AND\"],[29,\"BITWISE_XOR\"],\n");
        printf("    [30,\"COLON\"],[31,\"COMMA\"],[32,\"OPEN_PARENTHESIS\"],[33,\"CLOSE_PARENTHESIS\"],\n");
        printf("    [34,\"OPEN_SQUARE_BRACKET\"],[35,\"CLOSE_SQUARE_BRACKET\"],\n");
        printf("    [36,\"STRING\"],[37,\"NUMBER\"],[38,\"IDENTIFIER\"]\n");
        printf("  ],\n");

        printf("  \"symbol_tables\": {\n");

        printf("    \"identifiers\": [");
        for (i = 0; i < id_count; i++) {
            printf("[%d, ", identifiers[i].entry);
            print_json_string(identifiers[i].lexeme);
            printf("]%s", i < id_count - 1 ? ", " : "");
        }
        printf("],\n");

        printf("    \"numbers\": [");
        for (i = 0; i < num_count; i++) {
            printf("[%d, ", numbers[i].entry);
            print_json_string(numbers[i].lexeme);
            printf(", ");
            print_json_string(numbers[i].type);
            printf("]%s", i < num_count - 1 ? ", " : "");
        }
        printf("],\n");

        printf("    \"strings\": [");
        for (i = 0; i < str_count; i++) {
            printf("[%d, ", strings[i].entry);
            print_json_string(strings[i].lexeme);
            printf("]%s", i < str_count - 1 ? ", " : "");
        }
        printf("]\n");

        printf("  },\n");

        printf("  \"tokens\": [");
        for (i = 0; i < token_count; i++) {
            if (token_list[i].symbol_table_entry > 0)
                printf("[%d, %d]", token_list[i].token_id, token_list[i].symbol_table_entry);
            else
                printf("[%d]", token_list[i].token_id);
            if (i < token_count - 1) printf(", ");
        }
        printf("]\n");

        printf("}\n");
    }
%}

%option noyywrap

LETTER  [a-zA-Z_]
DIGIT   [0-9]
ID      {LETTER}({LETTER}|{DIGIT})*

INT     {DIGIT}+
FLOAT   ({DIGIT}+\.{DIGIT}+|{DIGIT}+\.|\.{DIGIT}+)([eE][+-]?{DIGIT}+)?
SCI     {DIGIT}+([eE][+-]?{DIGIT}+)
NUMBER  {FLOAT}|{SCI}|{INT}

STR_DQ  \"([^\\\n\"]|\\.)*\"
STR_SQ  \'([^\\\n\']|\\.)*\'

%%

[ \t\r\n]+  ; // Ignore whitespace

#[^\n]*     ; // Ignore comments (everything from # to end of line)

// --- Keywords (must come before the IDENTIFIER rule) ---
"def"       { add_token("DEF",    0); }
"if"        { add_token("IF",     0); }
"else"      { add_token("ELSE",   0); }
"for"       { add_token("FOR",    0); }
"while"     { add_token("WHILE",  0); }
"import"    { add_token("IMPORT", 0); }
"as"        { add_token("AS",     0); }
"in"        { add_token("IN",     0); }
"True"      { add_token("TRUE",   0); }
"False"     { add_token("FALSE",  0); }
"None"      { add_token("NONE",   0); }

// --- Compound operators (must come before single-character operators) ---
"//"        { add_token("FLOOR_DIVISION", 0); }
"=="        { add_token("DOUBLE_EQUAL",   0); }
"!="        { add_token("NOT_EQUAL",      0); }
"<="        { add_token("LESS_EQUAL",     0); }
">="        { add_token("GREATER_EQUAL",  0); }

// --- Single-character operators and delimiters ---
"."         { add_token("DOT",                  0); }
"@"         { add_token("AT_SIGN",              0); }
"+"         { add_token("PLUS_SIGN",            0); }
"-"         { add_token("MINUS_SIGN",           0); }
"*"         { add_token("MULTIPLICATION_SIGN",  0); }
"/"         { add_token("DIVISION_SIGN",        0); }
"%"         { add_token("MODULO",               0); }
">"         { add_token("GREATER_THAN",         0); }
"<"         { add_token("LESS_THAN",            0); }
"="         { add_token("EQUAL",                0); }
"&"         { add_token("BITWISE_AND",          0); }
"^"         { add_token("BITWISE_XOR",          0); }
":"         { add_token("COLON",                0); }
","         { add_token("COMMA",                0); }
"("         { add_token("OPEN_PARENTHESIS",     0); }
")"         { add_token("CLOSE_PARENTHESIS",    0); }
"["         { add_token("OPEN_SQUARE_BRACKET",  0); }
"]"         { add_token("CLOSE_SQUARE_BRACKET", 0); }

// --- Literals ---
{STR_DQ}    { add_token("STRING", add_string(yytext)); } // Double-quoted string
{STR_SQ}    { add_token("STRING", add_string(yytext)); } // Single-quoted string
{NUMBER}    { add_token("NUMBER", add_number(yytext)); }
{ID}        { add_token("IDENTIFIER", add_identifier(yytext)); }

// --- Error: unrecognized character ---
.           { fprintf(stderr, "Lexical error — unknown character: '%s'\n", yytext); }

%%

int main(int argc, char* argv[]) {
    const char* input_path;
    const char* output_path;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <input_file> [output_file.json]\n", argv[0]);
        return 1;
    }

    input_path  = argv[1];
    output_path = (argc == 3) ? argv[2] : "lexer_output.json";

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
