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

    typedef struct {
        int token_id;
        int symbol_table_entry;
    } TokenEntry;

    enum TokenId {
        TOK_COMMENT = 1,
        TOK_DEF = 2,
        TOK_IF = 3,
        TOK_ELSE = 4,
        TOK_FOR = 5,
        TOK_WHILE = 6,
        TOK_HASH_TAG = 7,
        TOK_DOT = 8,
        TOK_AT_SIGN = 9,
        TOK_PLUS_SIGN = 10,
        TOK_MINUS_SIGN = 11,
        TOK_MULTIPLICATION_SIGN = 12,
        TOK_DIVISION_SIGN = 13,
        TOK_MODULO = 14,
        TOK_GREATER_THAN = 15,
        TOK_LESS_THAN = 16,
        TOK_EQUAL = 17,
        TOK_SEMICOLON = 18,
        TOK_COMMA = 19,
        TOK_OPEN_PARENTHESIS = 20,
        TOK_CLOSE_PARENTHESIS = 21,
        TOK_OPEN_SQUARE_BRACKET = 22,
        TOK_CLOSE_SQUARE_BRACKET = 23,
        TOK_STRING = 24,
        TOK_NUMBER = 25,
        TOK_IDENTIFIER = 26
    };

    static int id_count = 0;
    static int num_count = 0;
    static int str_count = 0;

    static TokenEntry token_list[MAX_ENTRIES * 4];
    static int token_count = 0;

    static void print_json_string(const char* s) {
        const unsigned char* p = (const unsigned char*)s;
        putchar('"');
        while (*p) {
            switch (*p) {
                case '"': printf("\\\""); break;
                case '\\': printf("\\\\"); break;
                case '\b': printf("\\b"); break;
                case '\f': printf("\\f"); break;
                case '\n': printf("\\n"); break;
                case '\r': printf("\\r"); break;
                case '\t': printf("\\t"); break;
                default:
                    if (*p < 0x20) {
                        printf("\\u%04x", *p);
                    } else {
                        putchar(*p);
                    }
                    break;
            }
            p++;
        }
        putchar('"');
    }

    static int token_id_from_name(const char* token_name) {
        if (strcmp(token_name, "COMMENT") == 0) return TOK_COMMENT;
        if (strcmp(token_name, "DEF") == 0) return TOK_DEF;
        if (strcmp(token_name, "IF") == 0) return TOK_IF;
        if (strcmp(token_name, "ELSE") == 0) return TOK_ELSE;
        if (strcmp(token_name, "FOR") == 0) return TOK_FOR;
        if (strcmp(token_name, "WHILE") == 0) return TOK_WHILE;
        if (strcmp(token_name, "HASH_TAG") == 0) return TOK_HASH_TAG;
        if (strcmp(token_name, "DOT") == 0) return TOK_DOT;
        if (strcmp(token_name, "AT_SIGN") == 0) return TOK_AT_SIGN;
        if (strcmp(token_name, "PLUS_SIGN") == 0) return TOK_PLUS_SIGN;
        if (strcmp(token_name, "MINUS_SIGN") == 0) return TOK_MINUS_SIGN;
        if (strcmp(token_name, "MULTIPLICATION_SIGN") == 0) return TOK_MULTIPLICATION_SIGN;
        if (strcmp(token_name, "DIVISION_SIGN") == 0) return TOK_DIVISION_SIGN;
        if (strcmp(token_name, "MODULO") == 0) return TOK_MODULO;
        if (strcmp(token_name, "GREATER_THAN") == 0) return TOK_GREATER_THAN;
        if (strcmp(token_name, "LESS_THAN") == 0) return TOK_LESS_THAN;
        if (strcmp(token_name, "EQUAL") == 0) return TOK_EQUAL;
        if (strcmp(token_name, "SEMICOLON") == 0) return TOK_SEMICOLON;
        if (strcmp(token_name, "COMMA") == 0) return TOK_COMMA;
        if (strcmp(token_name, "OPEN_PARENTHESIS") == 0) return TOK_OPEN_PARENTHESIS;
        if (strcmp(token_name, "CLOSE_PARENTHESIS") == 0) return TOK_CLOSE_PARENTHESIS;
        if (strcmp(token_name, "OPEN_SQUARE_BRACKET") == 0) return TOK_OPEN_SQUARE_BRACKET;
        if (strcmp(token_name, "CLOSE_SQUARE_BRACKET") == 0) return TOK_CLOSE_SQUARE_BRACKET;
        if (strcmp(token_name, "STRING") == 0) return TOK_STRING;
        if (strcmp(token_name, "NUMBER") == 0) return TOK_NUMBER;
        if (strcmp(token_name, "IDENTIFIER") == 0) return TOK_IDENTIFIER;

        return 0;
    }

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
        size_t len;
        const char* start = lexeme;
        size_t copy_len;

        len = strlen(lexeme);
        if (len >= 2 && lexeme[0] == '"' && lexeme[len - 1] == '"') {
            start = lexeme + 1;
            copy_len = len - 2;
        } else {
            copy_len = len;
        }

        if (copy_len >= MAX_LEXEME_LEN) {
            copy_len = MAX_LEXEME_LEN - 1;
        }

        for (i = 0; i < str_count; i++) {
            if (strncmp(strings[i].lexeme, start, copy_len) == 0 && strings[i].lexeme[copy_len] == '\0') {
                return strings[i].entry;
            }
        }

        if (str_count >= MAX_ENTRIES) {
            fprintf(stderr, "String table overflow.\n");
            exit(1);
        }

        strings[str_count].entry = str_count + 1;
        strncpy(strings[str_count].lexeme, start, copy_len);
        strings[str_count].lexeme[copy_len] = '\0';
        str_count++;
        return str_count;
    }

    static void add_token(const char* token_name, int entry) {
        int token_id;

        if (token_count >= MAX_ENTRIES * 4) {
            fprintf(stderr, "Token list overflow.\n");
            exit(1);
        }

        token_id = token_id_from_name(token_name);
        if (token_id == 0) {
            fprintf(stderr, "Unknown token name: %s\n", token_name);
            exit(1);
        }

        token_list[token_count].token_id = token_id;
        token_list[token_count].symbol_table_entry = entry;
        token_count++;
    }

    static void print_results(void) {
        int i;

        printf("{\n");

        printf("  \"token_ids\": [\n");
        printf("    [1, \"COMMENT\"], [2, \"DEF\"], [3, \"IF\"], [4, \"ELSE\"], [5, \"FOR\"], [6, \"WHILE\"], [7, \"HASH_TAG\"], [8, \"DOT\"], [9, \"AT_SIGN\"], [10, \"PLUS_SIGN\"], [11, \"MINUS_SIGN\"], [12, \"MULTIPLICATION_SIGN\"], [13, \"DIVISION_SIGN\"], [14, \"MODULO\"], [15, \"GREATER_THAN\"], [16, \"LESS_THAN\"], [17, \"EQUAL\"], [18, \"SEMICOLON\"], [19, \"COMMA\"], [20, \"OPEN_PARENTHESIS\"], [21, \"CLOSE_PARENTHESIS\"], [22, \"OPEN_SQUARE_BRACKET\"], [23, \"CLOSE_SQUARE_BRACKET\"], [24, \"STRING\"], [25, \"NUMBER\"], [26, \"IDENTIFIER\"]\n");
        printf("  ],\n");

        printf("  \"symbol_tables\": {\n");

        printf("    \"identifiers\": [");
        for (i = 0; i < id_count; i++) {
            printf("[%d, ", identifiers[i].entry);
            print_json_string(identifiers[i].lexeme);
            printf("]");
            if (i < id_count - 1) {
                printf(", ");
            }
        }
        printf("],\n");

        printf("    \"numbers\": [");
        for (i = 0; i < num_count; i++) {
            printf("[%d, ", numbers[i].entry);
            print_json_string(numbers[i].lexeme);
            printf(", ");
            print_json_string(numbers[i].type);
            printf("]");
            if (i < num_count - 1) {
                printf(", ");
            }
        }
        printf("],\n");

        printf("    \"strings\": [");
        for (i = 0; i < str_count; i++) {
            printf("[%d, ", strings[i].entry);
            print_json_string(strings[i].lexeme);
            printf("]");
            if (i < str_count - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("  },\n");

        printf("  \"tokens\": [");
        for (i = 0; i < token_count; i++) {
            if (token_list[i].symbol_table_entry > 0) {
                printf("[%d, %d]", token_list[i].token_id, token_list[i].symbol_table_entry);
            } else {
                printf("[%d]", token_list[i].token_id);
            }

            if (i < token_count - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("}\n");
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

#[^\n]*                                          ;

"//"[^\n]*                                      { add_token("COMMENT", 0); }
"/*"([^*]|\*+[^*/])*\*+"/"                    { add_token("COMMENT", 0); }

"def"                                            { add_token("DEF", 0); }
"if"                                             { add_token("IF", 0); }
"else"                                           { add_token("ELSE", 0); }
"for"                                            { add_token("FOR", 0); }
"while"                                          { add_token("WHILE", 0); }

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