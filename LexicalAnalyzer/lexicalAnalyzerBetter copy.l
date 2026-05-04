%{
    // Lexical analyzer for the Triton language.
    // Input:  A Triton source file (.triton).
    // Output: A JSON file containing:
    //   - token_ids    : catalogue mapping each numeric ID to its name.
    //   - symbol_tables: identifiers, numbers (int/float/scientific), strings found.
    //   - tokens       : ordered token list; literals carry a symbol-table entry number.
    //
    // Standard library headers used:
    //   stdio.h  - printf / fprintf / freopen
    //   stdlib.h - exit
    //   string.h - strcmp / strncpy / strchr / strncmp / strlen

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    // ------------------------------------------------------------------
    // Compile-time limits
    // ------------------------------------------------------------------
    #define MAX_ENTRIES    4096   // max rows per symbol table
    #define MAX_LEXEME_LEN  256   // max characters per lexeme

    // ------------------------------------------------------------------
    // Symbol-table row structs
    // ------------------------------------------------------------------

    // One row of the identifier table.
    typedef struct {
        int  entry;
        char lexeme[MAX_LEXEME_LEN];
    } IdentifierEntry;

    // One row of the number table.
    // type is one of "int", "float", or "scientific".
    typedef struct {
        int  entry;
        char lexeme[MAX_LEXEME_LEN];
        char type[16];
    } NumberEntry;

    // One row of the string table (quotes are stripped before storing).
    typedef struct {
        int  entry;
        char lexeme[MAX_LEXEME_LEN];
    } StringEntry;

    static IdentifierEntry identifiers[MAX_ENTRIES];
    static NumberEntry     numbers[MAX_ENTRIES];
    static StringEntry     strings[MAX_ENTRIES];

    // ------------------------------------------------------------------
    // Token-list row struct
    // ------------------------------------------------------------------
    // symbol_table_entry is 0 for keywords/operators/delimiters, or a
    // positive index into the relevant symbol table for literals.
    typedef struct {
        int token_id;
        int symbol_table_entry;
    } TokenEntry;

    static TokenEntry token_list[MAX_ENTRIES * 8];

    // ------------------------------------------------------------------
    // Counters
    // ------------------------------------------------------------------
    static int id_count    = 0;
    static int num_count   = 0;
    static int str_count   = 0;
    static int token_count = 0;

    // ------------------------------------------------------------------
    // Token IDs  -  must match the catalogue printed in print_results()
    //               and the ID table in section 3.3 of the report.
    // ------------------------------------------------------------------
    enum TokenId {
        // Keywords (1-27)
        TOK_DEF       =  1,
        TOK_FOR       =  2,
        TOK_IF        =  3,
        TOK_ELSE      =  4,
        TOK_ELIF      =  5,
        TOK_WHILE     =  6,
        TOK_IMPORT    =  7,
        TOK_AS        =  8,
        TOK_IN        =  9,
        TOK_TRUE      = 10,
        TOK_FALSE     = 11,
        TOK_NONE      = 12,
        TOK_RETURN    = 13,
        TOK_BREAK     = 14,
        TOK_CONTINUE  = 15,
        TOK_PASS      = 16,
        TOK_AND       = 17,
        TOK_OR        = 18,
        TOK_NOT       = 19,
        TOK_IS        = 20,
        TOK_ASSERT    = 21,
        TOK_FROM      = 22,
        TOK_TRY       = 23,
        TOK_EXCEPT    = 24,
        TOK_GLOBAL    = 25,
        TOK_RAISE     = 26,
        TOK_DEL       = 27,

        // Identifiers and literals (28-32)
        TOK_IDENTIFIER = 28,
        TOK_SCIENTIFIC = 29,
        TOK_FLOAT      = 30,
        TOK_INT        = 31,
        TOK_STRING     = 32,

        // Arithmetic operators (33-39)
        TOK_PLUS       = 33,
        TOK_MINUS      = 34,
        TOK_MULTIPLY   = 35,
        TOK_DIVIDE     = 36,
        TOK_POWER      = 37,   // **
        TOK_FLOOR_DIV  = 38,   // //
        TOK_MODULO     = 39,

        // Comparison and assignment operators (40-46)
        TOK_EQUAL         = 40,   // =
        TOK_LESS_THAN     = 41,
        TOK_GREATER_THAN  = 42,
        TOK_NOT_EQUAL     = 43,   // !=
        TOK_LESS_EQUAL    = 44,   // <=
        TOK_GREATER_EQUAL = 45,   // >=
        TOK_DOUBLE_EQUAL  = 46,   // ==

        // Bitwise operators (47-52)
        TOK_BIT_AND    = 47,   // &
        TOK_BIT_OR     = 48,   // |
        TOK_BIT_NOT    = 49,   // ~
        TOK_BIT_XOR    = 50,   // ^
        TOK_LEFT_SHIFT = 51,   // <<
        TOK_RIGHT_SHIFT= 52,   // >>

        // Delimiters (53-62)
        TOK_DOT           = 53,
        TOK_COMMA         = 54,
        TOK_COLON         = 55,
        TOK_AT_SIGN       = 56,
        TOK_OPEN_PAREN    = 57,
        TOK_CLOSE_PAREN   = 58,
        TOK_OPEN_BRACKET  = 59,
        TOK_CLOSE_BRACKET = 60,
        TOK_OPEN_BRACE    = 61,
        TOK_CLOSE_BRACE   = 62
    };

    // ------------------------------------------------------------------
    // Helper: print a C string as a JSON string (escapes special chars)
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // Helper: map a token-name string to its numeric ID.
    // Returns 0 if the name is not recognized.
    // ------------------------------------------------------------------
    static int token_id_from_name(const char* name) {
        if (strcmp(name, "DEF")            == 0) return TOK_DEF;
        if (strcmp(name, "FOR")            == 0) return TOK_FOR;
        if (strcmp(name, "IF")             == 0) return TOK_IF;
        if (strcmp(name, "ELSE")           == 0) return TOK_ELSE;
        if (strcmp(name, "ELIF")           == 0) return TOK_ELIF;
        if (strcmp(name, "WHILE")          == 0) return TOK_WHILE;
        if (strcmp(name, "IMPORT")         == 0) return TOK_IMPORT;
        if (strcmp(name, "AS")             == 0) return TOK_AS;
        if (strcmp(name, "IN")             == 0) return TOK_IN;
        if (strcmp(name, "TRUE")           == 0) return TOK_TRUE;
        if (strcmp(name, "FALSE")          == 0) return TOK_FALSE;
        if (strcmp(name, "NONE")           == 0) return TOK_NONE;
        if (strcmp(name, "RETURN")         == 0) return TOK_RETURN;
        if (strcmp(name, "BREAK")          == 0) return TOK_BREAK;
        if (strcmp(name, "CONTINUE")       == 0) return TOK_CONTINUE;
        if (strcmp(name, "PASS")           == 0) return TOK_PASS;
        if (strcmp(name, "AND")            == 0) return TOK_AND;
        if (strcmp(name, "OR")             == 0) return TOK_OR;
        if (strcmp(name, "NOT")            == 0) return TOK_NOT;
        if (strcmp(name, "IS")             == 0) return TOK_IS;
        if (strcmp(name, "ASSERT")         == 0) return TOK_ASSERT;
        if (strcmp(name, "FROM")           == 0) return TOK_FROM;
        if (strcmp(name, "TRY")            == 0) return TOK_TRY;
        if (strcmp(name, "EXCEPT")         == 0) return TOK_EXCEPT;
        if (strcmp(name, "GLOBAL")         == 0) return TOK_GLOBAL;
        if (strcmp(name, "RAISE")          == 0) return TOK_RAISE;
        if (strcmp(name, "DEL")            == 0) return TOK_DEL;
        if (strcmp(name, "IDENTIFIER")     == 0) return TOK_IDENTIFIER;
        if (strcmp(name, "SCIENTIFIC")     == 0) return TOK_SCIENTIFIC;
        if (strcmp(name, "FLOAT")          == 0) return TOK_FLOAT;
        if (strcmp(name, "INT")            == 0) return TOK_INT;
        if (strcmp(name, "STRING")         == 0) return TOK_STRING;
        if (strcmp(name, "PLUS")           == 0) return TOK_PLUS;
        if (strcmp(name, "MINUS")          == 0) return TOK_MINUS;
        if (strcmp(name, "MULTIPLY")       == 0) return TOK_MULTIPLY;
        if (strcmp(name, "DIVIDE")         == 0) return TOK_DIVIDE;
        if (strcmp(name, "POWER")          == 0) return TOK_POWER;
        if (strcmp(name, "FLOOR_DIV")      == 0) return TOK_FLOOR_DIV;
        if (strcmp(name, "MODULO")         == 0) return TOK_MODULO;
        if (strcmp(name, "EQUAL")          == 0) return TOK_EQUAL;
        if (strcmp(name, "LESS_THAN")      == 0) return TOK_LESS_THAN;
        if (strcmp(name, "GREATER_THAN")   == 0) return TOK_GREATER_THAN;
        if (strcmp(name, "NOT_EQUAL")      == 0) return TOK_NOT_EQUAL;
        if (strcmp(name, "LESS_EQUAL")     == 0) return TOK_LESS_EQUAL;
        if (strcmp(name, "GREATER_EQUAL")  == 0) return TOK_GREATER_EQUAL;
        if (strcmp(name, "DOUBLE_EQUAL")   == 0) return TOK_DOUBLE_EQUAL;
        if (strcmp(name, "BIT_AND")        == 0) return TOK_BIT_AND;
        if (strcmp(name, "BIT_OR")         == 0) return TOK_BIT_OR;
        if (strcmp(name, "BIT_NOT")        == 0) return TOK_BIT_NOT;
        if (strcmp(name, "BIT_XOR")        == 0) return TOK_BIT_XOR;
        if (strcmp(name, "LEFT_SHIFT")     == 0) return TOK_LEFT_SHIFT;
        if (strcmp(name, "RIGHT_SHIFT")    == 0) return TOK_RIGHT_SHIFT;
        if (strcmp(name, "DOT")            == 0) return TOK_DOT;
        if (strcmp(name, "COMMA")          == 0) return TOK_COMMA;
        if (strcmp(name, "COLON")          == 0) return TOK_COLON;
        if (strcmp(name, "AT_SIGN")        == 0) return TOK_AT_SIGN;
        if (strcmp(name, "OPEN_PAREN")     == 0) return TOK_OPEN_PAREN;
        if (strcmp(name, "CLOSE_PAREN")    == 0) return TOK_CLOSE_PAREN;
        if (strcmp(name, "OPEN_BRACKET")   == 0) return TOK_OPEN_BRACKET;
        if (strcmp(name, "CLOSE_BRACKET")  == 0) return TOK_CLOSE_BRACKET;
        if (strcmp(name, "OPEN_BRACE")     == 0) return TOK_OPEN_BRACE;
        if (strcmp(name, "CLOSE_BRACE")    == 0) return TOK_CLOSE_BRACE;
        return 0;
    }

    // ------------------------------------------------------------------
    // Symbol-table insertion helpers.
    // Each function returns the entry number (1-based) of the lexeme,
    // inserting a new row only if the lexeme is not already present.
    // ------------------------------------------------------------------

    // Adds an identifier; returns its entry number.
    static int add_identifier(const char* lexeme) {
        int i;
        for (i = 0; i < id_count; i++)
            if (strcmp(identifiers[i].lexeme, lexeme) == 0) return identifiers[i].entry;
        if (id_count >= MAX_ENTRIES) { fprintf(stderr, "Identifier table overflow.\n"); exit(1); }
        identifiers[id_count].entry = id_count + 1;
        strncpy(identifiers[id_count].lexeme, lexeme, MAX_LEXEME_LEN - 1);
        identifiers[id_count].lexeme[MAX_LEXEME_LEN - 1] = '\0';
        return ++id_count;
    }

    // Adds a number with its explicit type tag; returns its entry number.
    static int add_number(const char* lexeme, const char* type_str) {
        int i;
        for (i = 0; i < num_count; i++)
            if (strcmp(numbers[i].lexeme, lexeme) == 0) return numbers[i].entry;
        if (num_count >= MAX_ENTRIES) { fprintf(stderr, "Number table overflow.\n"); exit(1); }
        numbers[num_count].entry = num_count + 1;
        strncpy(numbers[num_count].lexeme, lexeme, MAX_LEXEME_LEN - 1);
        numbers[num_count].lexeme[MAX_LEXEME_LEN - 1] = '\0';
        strncpy(numbers[num_count].type, type_str, 15);
        numbers[num_count].type[15] = '\0';
        return ++num_count;
    }

    // Adds a string after stripping surrounding quotes; returns its entry number.
    static int add_string(const char* lexeme) {
        int i;
        size_t len = strlen(lexeme);
        const char* start = lexeme;
        size_t copy_len = len;
        if (len >= 2 && (lexeme[0] == '"' || lexeme[0] == '\'') && lexeme[len-1] == lexeme[0]) {
            start    = lexeme + 1;
            copy_len = len - 2;
        }
        if (copy_len >= MAX_LEXEME_LEN) copy_len = MAX_LEXEME_LEN - 1;
        for (i = 0; i < str_count; i++)
            if (strncmp(strings[i].lexeme, start, copy_len) == 0 && strings[i].lexeme[copy_len] == '\0')
                return strings[i].entry;
        if (str_count >= MAX_ENTRIES) { fprintf(stderr, "String table overflow.\n"); exit(1); }
        strings[str_count].entry = str_count + 1;
        strncpy(strings[str_count].lexeme, start, copy_len);
        strings[str_count].lexeme[copy_len] = '\0';
        return ++str_count;
    }

    // ------------------------------------------------------------------
    // Token-list builder.
    // entry = 0  for keywords, operators, and delimiters.
    // entry > 0  for identifiers / numbers / strings.
    // ------------------------------------------------------------------
    static void add_token(const char* token_name, int entry) {
        int id;
        if (token_count >= MAX_ENTRIES * 8) { fprintf(stderr, "Token list overflow.\n"); exit(1); }
        id = token_id_from_name(token_name);
        if (id == 0) { fprintf(stderr, "Unknown token name: %s\n", token_name); exit(1); }
        token_list[token_count].token_id          = id;
        token_list[token_count].symbol_table_entry = entry;
        token_count++;
    }

    // ------------------------------------------------------------------
    // print_results: writes the complete JSON output to stdout.
    // ------------------------------------------------------------------
    static void print_results(void) {
        int i;
        printf("{\n");

        // Token ID catalogue
        printf("  \"token_ids\": [\n");
        printf("    [1,\"DEF\"],[2,\"FOR\"],[3,\"IF\"],[4,\"ELSE\"],[5,\"ELIF\"],[6,\"WHILE\"],\n");
        printf("    [7,\"IMPORT\"],[8,\"AS\"],[9,\"IN\"],[10,\"TRUE\"],[11,\"FALSE\"],[12,\"NONE\"],\n");
        printf("    [13,\"RETURN\"],[14,\"BREAK\"],[15,\"CONTINUE\"],[16,\"PASS\"],\n");
        printf("    [17,\"AND\"],[18,\"OR\"],[19,\"NOT\"],[20,\"IS\"],[21,\"ASSERT\"],\n");
        printf("    [22,\"FROM\"],[23,\"TRY\"],[24,\"EXCEPT\"],[25,\"GLOBAL\"],[26,\"RAISE\"],[27,\"DEL\"],\n");
        printf("    [28,\"IDENTIFIER\"],[29,\"SCIENTIFIC\"],[30,\"FLOAT\"],[31,\"INT\"],[32,\"STRING\"],\n");
        printf("    [33,\"PLUS\"],[34,\"MINUS\"],[35,\"MULTIPLY\"],[36,\"DIVIDE\"],[37,\"POWER\"],[38,\"FLOOR_DIV\"],[39,\"MODULO\"],\n");
        printf("    [40,\"EQUAL\"],[41,\"LESS_THAN\"],[42,\"GREATER_THAN\"],[43,\"NOT_EQUAL\"],[44,\"LESS_EQUAL\"],[45,\"GREATER_EQUAL\"],[46,\"DOUBLE_EQUAL\"],\n");
        printf("    [47,\"BIT_AND\"],[48,\"BIT_OR\"],[49,\"BIT_NOT\"],[50,\"BIT_XOR\"],[51,\"LEFT_SHIFT\"],[52,\"RIGHT_SHIFT\"],\n");
        printf("    [53,\"DOT\"],[54,\"COMMA\"],[55,\"COLON\"],[56,\"AT_SIGN\"],\n");
        printf("    [57,\"OPEN_PAREN\"],[58,\"CLOSE_PAREN\"],[59,\"OPEN_BRACKET\"],[60,\"CLOSE_BRACKET\"],[61,\"OPEN_BRACE\"],[62,\"CLOSE_BRACE\"]\n");
        printf("  ],\n");

        // Symbol tables
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

        // Token list
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

%option noyywrap yylineno

LETTER  [a-zA-Z_]
DIGIT   [0-9]
ID      {LETTER}({LETTER}|{DIGIT})*

SCI     {DIGIT}+(\.[0-9]+)?[eE][+-]?{DIGIT}+
FLOAT   {DIGIT}+\.{DIGIT}+|{DIGIT}+\.|\.{DIGIT}+
INT     {DIGIT}+

STR_DQ  \"([^\\\n\"]|\\.)*\"
STR_SQ  \'([^\\\n\']|\\.)*\'

%%

[ \t\r\n]+  ;   /* Whitespace is skipped - not a token in Triton */

#[^\n]*     ;   /* Comment: everything from # to end of line is ignored */

/* -----------------------------------------------------------------
   Keywords - listed before the IDENTIFIER rule so Flex gives them
   priority over the general identifier pattern.
   ----------------------------------------------------------------- */
"def"       { add_token("DEF",      0); }
"for"       { add_token("FOR",      0); }
"if"        { add_token("IF",       0); }
"else"      { add_token("ELSE",     0); }
"elif"      { add_token("ELIF",     0); }
"while"     { add_token("WHILE",    0); }
"import"    { add_token("IMPORT",   0); }
"as"        { add_token("AS",       0); }
"in"        { add_token("IN",       0); }
"True"      { add_token("TRUE",     0); }
"False"     { add_token("FALSE",    0); }
"None"      { add_token("NONE",     0); }
"return"    { add_token("RETURN",   0); }
"break"     { add_token("BREAK",    0); }
"continue"  { add_token("CONTINUE", 0); }
"pass"      { add_token("PASS",     0); }
"and"       { add_token("AND",      0); }
"or"        { add_token("OR",       0); }
"not"       { add_token("NOT",      0); }
"is"        { add_token("IS",       0); }
"assert"    { add_token("ASSERT",   0); }
"from"      { add_token("FROM",     0); }
"try"       { add_token("TRY",      0); }
"except"    { add_token("EXCEPT",   0); }
"global"    { add_token("GLOBAL",   0); }
"raise"     { add_token("RAISE",    0); }
"del"       { add_token("DEL",      0); }

/* -----------------------------------------------------------------
   Compound operators - must come before their single-char prefixes
   so that Flex applies maximal munch correctly.
   Ex: "**" before "*", "//" before "/", "<<" before "<".
   ----------------------------------------------------------------- */
"**"        { add_token("POWER",         0); }
"//"        { add_token("FLOOR_DIV",     0); }
"<<"        { add_token("LEFT_SHIFT",    0); }
">>"        { add_token("RIGHT_SHIFT",   0); }
"=="        { add_token("DOUBLE_EQUAL",  0); }
"!="        { add_token("NOT_EQUAL",     0); }
"<="        { add_token("LESS_EQUAL",    0); }
">="        { add_token("GREATER_EQUAL", 0); }

/* -----------------------------------------------------------------
   Single-character operators
   ----------------------------------------------------------------- */
"+"         { add_token("PLUS",         0); }
"-"         { add_token("MINUS",        0); }
"*"         { add_token("MULTIPLY",     0); }
"/"         { add_token("DIVIDE",       0); }
"%"         { add_token("MODULO",       0); }
"="         { add_token("EQUAL",        0); }
"<"         { add_token("LESS_THAN",    0); }
">"         { add_token("GREATER_THAN", 0); }
"&"         { add_token("BIT_AND",      0); }
"|"         { add_token("BIT_OR",       0); }
"~"         { add_token("BIT_NOT",      0); }
"^"         { add_token("BIT_XOR",      0); }

/* -----------------------------------------------------------------
   Delimiters
   ----------------------------------------------------------------- */
"."         { add_token("DOT",           0); }
","         { add_token("COMMA",         0); }
":"         { add_token("COLON",         0); }
"@"         { add_token("AT_SIGN",       0); }
"("         { add_token("OPEN_PAREN",    0); }
")"         { add_token("CLOSE_PAREN",   0); }
"["         { add_token("OPEN_BRACKET",  0); }
"]"         { add_token("CLOSE_BRACKET", 0); }
"{"         { add_token("OPEN_BRACE",    0); }
"}"         { add_token("CLOSE_BRACE",   0); }

/* -----------------------------------------------------------------
   Literals
   SCI and FLOAT must come before INT so Flex does not consume only
   the integer prefix of a float or scientific number.
   ----------------------------------------------------------------- */
{STR_DQ}    { add_token("STRING",     add_string(yytext));              }
{STR_SQ}    { add_token("STRING",     add_string(yytext));              }
{SCI}       { add_token("SCIENTIFIC", add_number(yytext, "scientific")); }
{FLOAT}     { add_token("FLOAT",      add_number(yytext, "float"));     }
{INT}       { add_token("INT",        add_number(yytext, "int"));       }
{ID}        { add_token("IDENTIFIER", add_identifier(yytext));          }

/* -----------------------------------------------------------------
   Error rule - catches any character not matched by the rules above.
   Reports the offending character and continues scanning.
   ----------------------------------------------------------------- */
.           { fprintf(stderr, "Lexical error - unrecognized character: '%s' at line %d\n",
                      yytext, yylineno); }

%%

/* -----------------------------------------------------------------
   main: opens input/output files, runs the lexer, writes JSON.
   Usage: ./lexer <input_file.triton> [output_file.json]
   ----------------------------------------------------------------- */
int main(int argc, char* argv[]) {
    const char* input_path;
    const char* output_path;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <input_file.triton> [output_file.json]\n", argv[0]);
        return 1;
    }

    input_path  = argv[1];
    output_path = (argc == 3) ? argv[2] : "lexer_output.json";

    if (freopen(input_path,  "r", stdin)  == NULL) {
        fprintf(stderr, "Error: could not open input file: %s\n", input_path);
        return 1;
    }
    if (freopen(output_path, "w", stdout) == NULL) {
        fprintf(stderr, "Error: could not open output file: %s\n", output_path);
        return 1;
    }

    yylex();           /* Run the lexer - fills symbol tables and token list */
    print_results();   /* Write JSON output to the redirected stdout          */

    fprintf(stderr, "Done. Output written to: %s\n", output_path);
    return 0;
}
