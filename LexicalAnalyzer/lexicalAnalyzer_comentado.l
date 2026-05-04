/* ============================================================
   ANALIZADOR LÉXICO PARA EL LENGUAJE TRITON
   ============================================================
   ¿Qué hace este archivo?
   Este archivo es el CORAZÓN del proyecto. Está escrito en
   el lenguaje "Flex" (.l), que es una herramienta que lee
   las reglas que tú defines aquí y genera automáticamente
   un programa en C capaz de leer código Triton y separarlo
   en "tokens" (piezas con significado).

   Un archivo Flex tiene TRES secciones separadas por "%%":
     SECCIÓN 1 (entre %{ y %} + definiciones): código C y
                variables que se van a usar en todo el programa.
     SECCIÓN 2 (entre %% y %%): las REGLAS. Cada regla dice
                "si ves este patrón, haz esta acción".
     SECCIÓN 3 (después del segundo %%): el main() y funciones
                adicionales de C.

   ENTRADA:  Un archivo de código Triton (.triton)
   SALIDA:   Un archivo JSON con tres cosas:
               - token_ids    : catálogo de qué número le toca a qué token
               - symbol_tables: tablas de identificadores, números y strings
               - tokens       : la lista de tokens del código analizado
   ============================================================ */


/* ============================================================
   SECCIÓN 1 — CÓDIGO C Y DEFINICIONES
   Todo lo que esté entre %{ y %} se copia tal cual al
   archivo C que Flex genera. Aquí ponemos: includes,
   structs, variables globales y funciones auxiliares.
   ============================================================ */
%{

 /* ------------------------------------------------------------
   INCLUDES — librerías estándar de C que necesitamos
   ------------------------------------------------------------ */
    #include <stdio.h>   /* Para printf(), fprintf(), freopen() — imprimir y abrir archivos */
    #include <stdlib.h>  /* Para exit() — salir del programa con un código de error         */
    #include <string.h>  /* Para strcmp(), strncpy(), strlen(), etc. — manipular strings    */


 /* ------------------------------------------------------------
   CONSTANTES — límites máximos del programa
   ------------------------------------------------------------ */
    /* Número máximo de filas que puede tener cada tabla de símbolos.
       Si el código Triton tiene más de 4096 identificadores distintos,
       el programa mostrará un error. En la práctica nunca se llega a esto. */
    #define MAX_ENTRIES    4096

    /* Longitud máxima en caracteres que puede tener un lexema (una palabra).
       Por ejemplo, un nombre de variable no puede tener más de 255 letras. */
    #define MAX_LEXEME_LEN  256


 /* ------------------------------------------------------------
   STRUCTS — las "plantillas" de cada fila de las tablas

   Un struct en C es como un molde para guardar varios datos
   juntos. Aquí definimos cómo se ve cada fila de cada tabla.
   ------------------------------------------------------------ */

    /* Plantilla para UNA FILA de la tabla de identificadores.
       Un identificador es cualquier nombre de variable o función,
       por ejemplo: "x", "mi_variable", "softmax_kernel" */
    typedef struct {
        int  entry;                  /* Número de entrada en la tabla, empieza en 1 */
        char lexeme[MAX_LEXEME_LEN]; /* El texto del identificador, ej: "mi_variable" */
    } IdentifierEntry;               /* Le ponemos el nombre "IdentifierEntry" a este molde */

    /* Plantilla para UNA FILA de la tabla de números.
       Guardamos el número tal como aparece en el código y su tipo.
       Ejemplos: 42 (int), 3.14 (float), 1e-5 (scientific) */
    typedef struct {
        int  entry;                  /* Número de entrada en la tabla, empieza en 1   */
        char lexeme[MAX_LEXEME_LEN]; /* El texto del número, ej: "3.14" o "1e-05"     */
        char type[16];               /* El tipo: "int", "float" o "scientific"         */
    } NumberEntry;

    /* Plantilla para UNA FILA de la tabla de strings.
       Un string es texto entre comillas, ej: "hola" o 'mundo'.
       Las comillas se quitan antes de guardarlo. */
    typedef struct {
        int  entry;                  /* Número de entrada en la tabla, empieza en 1   */
        char lexeme[MAX_LEXEME_LEN]; /* El texto sin comillas, ej: hola               */
    } StringEntry;


 /* ------------------------------------------------------------
   TABLAS DE SÍMBOLOS — los arreglos donde guardamos los datos

   Cada tabla es un arreglo de su struct correspondiente.
   "static" significa que la variable solo existe dentro de
   este archivo; nadie de afuera puede tocarla.
   ------------------------------------------------------------ */
    static IdentifierEntry identifiers[MAX_ENTRIES]; /* Tabla de identificadores: hasta 4096 filas */
    static NumberEntry     numbers[MAX_ENTRIES];     /* Tabla de números: hasta 4096 filas         */
    static StringEntry     strings[MAX_ENTRIES];     /* Tabla de strings: hasta 4096 filas         */


 /* ------------------------------------------------------------
   STRUCT PARA LA LISTA DE TOKENS

   Además de las tablas de símbolos, necesitamos guardar la
   SECUENCIA de tokens en el orden en que aparecen en el código.
   Cada token en la lista tiene dos datos:
     - token_id: el número que identifica qué tipo de token es
     - symbol_table_entry: si es un literal (número, string,
       identificador), apunta a su fila en la tabla.
       Si es keyword u operador, este valor es 0.

   Ejemplo: el token "x" sería [28, 1] — tipo IDENTIFIER (28),
   entrada 1 en la tabla de identificadores.
   El token "+" sería [33] — tipo PLUS (33), sin entrada en tabla.
   ------------------------------------------------------------ */
    typedef struct {
        int token_id;            /* Número que identifica el tipo de token (ej: 33 = PLUS)     */
        int symbol_table_entry;  /* Entrada en tabla de símbolos; 0 si no aplica               */
    } TokenEntry;

    /* La lista de tokens: un arreglo que puede guardar hasta 4096*8 = 32768 tokens.
       Multiplicamos por 8 porque un archivo de código puede tener muchos más tokens
       que identificadores (cada línea tiene varios tokens). */
    static TokenEntry token_list[MAX_ENTRIES * 8];


 /* ------------------------------------------------------------
   CONTADORES — llevan la cuenta de cuántas filas hay en cada tabla
   ------------------------------------------------------------ */
    static int id_count    = 0; /* Cuántos identificadores distintos se han encontrado */
    static int num_count   = 0; /* Cuántos números distintos se han encontrado         */
    static int str_count   = 0; /* Cuántos strings distintos se han encontrado         */
    static int token_count = 0; /* Cuántos tokens en total se han encontrado           */


 /* ------------------------------------------------------------
   ENUM DE IDs DE TOKENS

   Un "enum" en C es una lista de constantes con nombre.
   Aquí le asignamos un número único a cada tipo de token.
   Esto es importante porque el JSON de salida usa números,
   no palabras — es más compacto y fácil de procesar.

   Ejemplo: en vez de escribir "PLUS" en el JSON,
   escribimos 33. El catálogo token_ids explica que 33 = PLUS.
   ------------------------------------------------------------ */
    enum TokenId {

        /* --- PALABRAS RESERVADAS (1 al 27) ---
           Son palabras que el lenguaje ya tiene definidas y no
           se pueden usar como nombres de variables. */
        TOK_DEF       =  1,  /* def      — para declarar funciones                */
        TOK_FOR       =  2,  /* for      — bucle for                               */
        TOK_IF        =  3,  /* if       — condicional                             */
        TOK_ELSE      =  4,  /* else     — rama alternativa del if                 */
        TOK_ELIF      =  5,  /* elif     — else if encadenado                      */
        TOK_WHILE     =  6,  /* while    — bucle while                             */
        TOK_IMPORT    =  7,  /* import   — importar módulos                        */
        TOK_AS        =  8,  /* as       — alias en imports (import triton as tl)  */
        TOK_IN        =  9,  /* in       — pertenencia (for x in lista)            */
        TOK_TRUE      = 10,  /* True     — valor booleano verdadero                */
        TOK_FALSE     = 11,  /* False    — valor booleano falso                    */
        TOK_NONE      = 12,  /* None     — valor nulo                              */
        TOK_RETURN    = 13,  /* return   — regresar un valor de una función        */
        TOK_BREAK     = 14,  /* break    — salir de un bucle                       */
        TOK_CONTINUE  = 15,  /* continue — saltar a la siguiente iteración         */
        TOK_PASS      = 16,  /* pass     — no hacer nada (bloque vacío)            */
        TOK_AND       = 17,  /* and      — operador lógico Y                       */
        TOK_OR        = 18,  /* or       — operador lógico O                       */
        TOK_NOT       = 19,  /* not      — operador lógico negación                */
        TOK_IS        = 20,  /* is       — comparar identidad de objetos           */
        TOK_ASSERT    = 21,  /* assert   — verificar una condición                 */
        TOK_FROM      = 22,  /* from     — from X import Y                         */
        TOK_TRY       = 23,  /* try      — bloque try/except                       */
        TOK_EXCEPT    = 24,  /* except   — capturar errores                        */
        TOK_GLOBAL    = 25,  /* global   — declarar variable global                */
        TOK_RAISE     = 26,  /* raise    — lanzar un error                         */
        TOK_DEL       = 27,  /* del      — eliminar una variable                   */

        /* --- IDENTIFICADORES Y LITERALES (28 al 32) ---
           Son los tokens que SÍ guardan su valor en la tabla de símbolos. */
        TOK_IDENTIFIER = 28, /* Nombres de variables y funciones, ej: x, mi_var   */
        TOK_SCIENTIFIC = 29, /* Números en notación científica, ej: 1e-5, 3.14e2  */
        TOK_FLOAT      = 30, /* Números con punto decimal, ej: 3.14, 0.5          */
        TOK_INT        = 31, /* Números enteros, ej: 42, 0, 100                   */
        TOK_STRING     = 32, /* Texto entre comillas, ej: "hola" o 'mundo'        */

        /* --- OPERADORES ARITMÉTICOS (33 al 39) --- */
        TOK_PLUS       = 33, /* +   suma                                           */
        TOK_MINUS      = 34, /* -   resta                                          */
        TOK_MULTIPLY   = 35, /* *   multiplicación                                 */
        TOK_DIVIDE     = 36, /* /   división                                       */
        TOK_POWER      = 37, /* **  potencia (2**3 = 8)                            */
        TOK_FLOOR_DIV  = 38, /* //  división entera (7//2 = 3)                     */
        TOK_MODULO     = 39, /* %   módulo / residuo (7%2 = 1)                     */

        /* --- OPERADORES DE COMPARACIÓN Y ASIGNACIÓN (40 al 46) --- */
        TOK_EQUAL         = 40, /* =   asignación (x = 5)                          */
        TOK_LESS_THAN     = 41, /* <   menor que                                   */
        TOK_GREATER_THAN  = 42, /* >   mayor que                                   */
        TOK_NOT_EQUAL     = 43, /* !=  diferente de                                */
        TOK_LESS_EQUAL    = 44, /* <=  menor o igual que                           */
        TOK_GREATER_EQUAL = 45, /* >=  mayor o igual que                           */
        TOK_DOUBLE_EQUAL  = 46, /* ==  comparación de igualdad (x == 5)            */

        /* --- OPERADORES DE BITS (47 al 52) ---
           Operan directamente sobre los bits de un número. */
        TOK_BIT_AND    = 47, /* &   AND a nivel de bits (0b1010 & 0b1100 = 0b1000) */
        TOK_BIT_OR     = 48, /* |   OR a nivel de bits  (0b1010 | 0b0101 = 0b1111) */
        TOK_BIT_NOT    = 49, /* ~   NOT a nivel de bits (invierte todos los bits)   */
        TOK_BIT_XOR    = 50, /* ^   XOR a nivel de bits (diferencia bit a bit)      */
        TOK_LEFT_SHIFT = 51, /* <<  desplazamiento de bits a la izquierda           */
        TOK_RIGHT_SHIFT= 52, /* >>  desplazamiento de bits a la derecha             */

        /* --- DELIMITADORES (53 al 62) ---
           Símbolos que dan estructura al código pero no calculan nada. */
        TOK_DOT           = 53, /* .  acceso a atributo (tl.load)                  */
        TOK_COMMA         = 54, /* ,  separador de argumentos                      */
        TOK_COLON         = 55, /* :  inicio de bloque (def f():)                  */
        TOK_AT_SIGN       = 56, /* @  decorador (@triton.jit)                      */
        TOK_OPEN_PAREN    = 57, /* (  abre paréntesis                              */
        TOK_CLOSE_PAREN   = 58, /* )  cierra paréntesis                            */
        TOK_OPEN_BRACKET  = 59, /* [  abre corchete (listas/índices)               */
        TOK_CLOSE_BRACKET = 60, /* ]  cierra corchete                              */
        TOK_OPEN_BRACE    = 61, /* {  abre llave (diccionarios)                    */
        TOK_CLOSE_BRACE   = 62  /* }  cierra llave                                 */
    };


/* ============================================================
   FUNCIONES AUXILIARES
   Son funciones de C que usamos dentro de las reglas de Flex.
   Se declaran aquí (en la sección 1) para que estén disponibles
   cuando Flex las llame desde la sección 2.
   ============================================================ */

 /* ------------------------------------------------------------
   FUNCIÓN: print_json_string
   ¿Qué hace?
     Imprime un texto de C como si fuera un string JSON válido,
     es decir, rodeado de comillas dobles y con los caracteres
     especiales escapados correctamente.

   ¿Por qué es necesaria?
     En JSON, si un texto contiene una comilla doble " o una
     barra invertida \, hay que escribirlos como \" y \\.
     Si no hacemos esto, el JSON resultante sería inválido.

   Ejemplo:
     Entrada C:  hola "mundo"
     Salida JSON: "hola \"mundo\""
   ------------------------------------------------------------ */
    static void print_json_string(const char* s) {
        /* Tratamos el string como bytes sin signo para manejar
           correctamente caracteres especiales y acentos */
        const unsigned char* p = (const unsigned char*)s;

        putchar('"'); /* Imprime la comilla de apertura del string JSON */

        /* Recorremos el string carácter por carácter hasta el final (\0) */
        while (*p) {
            switch (*p) {
                /* Cada case maneja un carácter especial que hay que escapar en JSON */
                case '"':  printf("\\\""); break; /* " → \" */
                case '\\': printf("\\\\"); break; /* \ → \\ */
                case '\b': printf("\\b");  break; /* backspace */
                case '\f': printf("\\f");  break; /* form feed */
                case '\n': printf("\\n");  break; /* salto de línea */
                case '\r': printf("\\r");  break; /* retorno de carro */
                case '\t': printf("\\t");  break; /* tabulación */
                default:
                    /* Si el carácter es un control (código < 32) lo imprimimos
                       como \uXXXX (formato unicode de JSON) */
                    if (*p < 0x20) printf("\\u%04x", *p);
                    /* Si es un carácter normal, lo imprimimos tal cual */
                    else           putchar(*p);
                    break;
            }
            p++; /* Avanzamos al siguiente carácter del string */
        }

        putchar('"'); /* Imprime la comilla de cierre del string JSON */
    }


/* ------------------------------------------------------------
   FUNCIÓN: token_id_from_name
   ¿Qué hace?
     Recibe el NOMBRE de un token como string (ej: "PLUS") y
     devuelve su NÚMERO correspondiente (ej: 33).

   ¿Por qué es necesaria?
     Las reglas de Flex llaman add_token("PLUS", 0).
     add_token necesita saber que "PLUS" = 33 para guardarlo
     en la lista. Esta función hace esa conversión.

   Devuelve 0 si el nombre no se reconoce (error).
   ------------------------------------------------------------ */
    static int token_id_from_name(const char* name) {
        /* strcmp compara dos strings. Devuelve 0 si son iguales.
           Por eso la condición es "== 0" para saber si coinciden. */

        /* Comparamos el nombre recibido con cada nombre posible
           y devolvemos el ID correspondiente del enum */

        /* Palabras reservadas */
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

        /* Identificadores y literales */
        if (strcmp(name, "IDENTIFIER")     == 0) return TOK_IDENTIFIER;
        if (strcmp(name, "SCIENTIFIC")     == 0) return TOK_SCIENTIFIC;
        if (strcmp(name, "FLOAT")          == 0) return TOK_FLOAT;
        if (strcmp(name, "INT")            == 0) return TOK_INT;
        if (strcmp(name, "STRING")         == 0) return TOK_STRING;

        /* Operadores aritméticos */
        if (strcmp(name, "PLUS")           == 0) return TOK_PLUS;
        if (strcmp(name, "MINUS")          == 0) return TOK_MINUS;
        if (strcmp(name, "MULTIPLY")       == 0) return TOK_MULTIPLY;
        if (strcmp(name, "DIVIDE")         == 0) return TOK_DIVIDE;
        if (strcmp(name, "POWER")          == 0) return TOK_POWER;
        if (strcmp(name, "FLOOR_DIV")      == 0) return TOK_FLOOR_DIV;
        if (strcmp(name, "MODULO")         == 0) return TOK_MODULO;

        /* Operadores de comparación */
        if (strcmp(name, "EQUAL")          == 0) return TOK_EQUAL;
        if (strcmp(name, "LESS_THAN")      == 0) return TOK_LESS_THAN;
        if (strcmp(name, "GREATER_THAN")   == 0) return TOK_GREATER_THAN;
        if (strcmp(name, "NOT_EQUAL")      == 0) return TOK_NOT_EQUAL;
        if (strcmp(name, "LESS_EQUAL")     == 0) return TOK_LESS_EQUAL;
        if (strcmp(name, "GREATER_EQUAL")  == 0) return TOK_GREATER_EQUAL;
        if (strcmp(name, "DOUBLE_EQUAL")   == 0) return TOK_DOUBLE_EQUAL;

        /* Operadores de bits */
        if (strcmp(name, "BIT_AND")        == 0) return TOK_BIT_AND;
        if (strcmp(name, "BIT_OR")         == 0) return TOK_BIT_OR;
        if (strcmp(name, "BIT_NOT")        == 0) return TOK_BIT_NOT;
        if (strcmp(name, "BIT_XOR")        == 0) return TOK_BIT_XOR;
        if (strcmp(name, "LEFT_SHIFT")     == 0) return TOK_LEFT_SHIFT;
        if (strcmp(name, "RIGHT_SHIFT")    == 0) return TOK_RIGHT_SHIFT;

        /* Delimitadores */
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

        return 0; /* Si no se reconoció el nombre, devolvemos 0 (error) */
    }


/* ------------------------------------------------------------
   FUNCIÓN: add_identifier
   ¿Qué hace?
     Agrega un identificador a la tabla de identificadores
     y devuelve su número de entrada.
     Si el identificador ya existe, devuelve su entrada existente
     (no lo duplica).

   Ejemplo:
     Primera vez que aparece "x"  → lo agrega, devuelve 1
     Segunda vez que aparece "x"  → ya existe, devuelve 1
     Primera vez que aparece "y"  → lo agrega, devuelve 2
   ------------------------------------------------------------ */
    static int add_identifier(const char* lexeme) {
        int i;

        /* Recorremos la tabla buscando si el lexema ya existe */
        for (i = 0; i < id_count; i++)
            if (strcmp(identifiers[i].lexeme, lexeme) == 0)
                return identifiers[i].entry; /* Ya existe, devolvemos su número */

        /* Si llegamos aquí, el lexema es nuevo */

        /* Verificamos que la tabla no esté llena */
        if (id_count >= MAX_ENTRIES) {
            fprintf(stderr, "Error: tabla de identificadores llena.\n");
            exit(1); /* Terminamos el programa con error */
        }

        /* Guardamos el nuevo identificador en la siguiente posición libre */
        identifiers[id_count].entry = id_count + 1; /* Las entradas empiezan en 1, no en 0 */

        /* strncpy copia el texto del lexema a la tabla.
           El -1 es para dejar espacio al carácter de fin de string '\0' */
        strncpy(identifiers[id_count].lexeme, lexeme, MAX_LEXEME_LEN - 1);

        /* Nos aseguramos de que el string termine correctamente */
        identifiers[id_count].lexeme[MAX_LEXEME_LEN - 1] = '\0';

        /* Incrementamos el contador y devolvemos el número de entrada */
        return ++id_count; /* ++ antes significa: primero suma 1, luego devuelve el valor */
    }


/* ------------------------------------------------------------
   FUNCIÓN: add_number
   ¿Qué hace?
     Igual que add_identifier pero para números.
     Además recibe el tipo del número ("int", "float" o "scientific")
     para guardarlo en la columna "type" de la tabla.

   El tipo se decide en la sección de reglas (sección 2),
   no aquí, porque Flex sabe cuál regla disparó.
   ------------------------------------------------------------ */
    static int add_number(const char* lexeme, const char* type_str) {
        int i;

        /* Buscamos si este número exacto ya fue visto antes */
        for (i = 0; i < num_count; i++)
            if (strcmp(numbers[i].lexeme, lexeme) == 0) return numbers[i].entry;

        /* Si es nuevo, verificamos que la tabla no esté llena */
        if (num_count >= MAX_ENTRIES) {
            fprintf(stderr, "Error: tabla de números llena.\n");
            exit(1);
        }

        /* Guardamos el número y su tipo en la tabla */
        numbers[num_count].entry = num_count + 1;
        strncpy(numbers[num_count].lexeme, lexeme, MAX_LEXEME_LEN - 1);
        numbers[num_count].lexeme[MAX_LEXEME_LEN - 1] = '\0';
        strncpy(numbers[num_count].type, type_str, 15); /* El campo type tiene solo 16 chars */
        numbers[num_count].type[15] = '\0';

        return ++num_count;
    }


/* ------------------------------------------------------------
   FUNCIÓN: add_string
   ¿Qué hace?
     Igual que add_identifier pero para strings.
     Además, QUITA las comillas antes de guardar el texto.

   ¿Por qué quitar las comillas?
     El código fuente tiene: "hola mundo"
     Pero en la tabla de símbolos queremos guardar: hola mundo
     Las comillas son parte de la sintaxis, no del valor.

   Funciona tanto con comillas dobles " como simples '
   ------------------------------------------------------------ */
    static int add_string(const char* lexeme) {
        int i;
        size_t len = strlen(lexeme); /* Longitud total del string incluyendo comillas */
        const char* start = lexeme;  /* Por defecto empezamos desde el principio      */
        size_t copy_len = len;       /* Por defecto copiamos todo                     */

        /* Si el string empieza y termina con el mismo tipo de comilla,
           ajustamos para saltarnos las comillas */
        if (len >= 2 &&
            (lexeme[0] == '"' || lexeme[0] == '\'') && /* Comienza con " o ' */
            lexeme[len-1] == lexeme[0]) {               /* Termina con el mismo tipo */
            start    = lexeme + 1;  /* Empezamos un carácter después de la comilla inicial */
            copy_len = len - 2;     /* Copiamos dos caracteres menos (quitamos ambas comillas) */
        }

        /* Nos aseguramos de no pasarnos del límite */
        if (copy_len >= MAX_LEXEME_LEN) copy_len = MAX_LEXEME_LEN - 1;

        /* Buscamos si este string ya existe en la tabla */
        for (i = 0; i < str_count; i++)
            if (strncmp(strings[i].lexeme, start, copy_len) == 0 &&
                strings[i].lexeme[copy_len] == '\0')
                return strings[i].entry;

        /* Si es nuevo, lo guardamos */
        if (str_count >= MAX_ENTRIES) {
            fprintf(stderr, "Error: tabla de strings llena.\n");
            exit(1);
        }
        strings[str_count].entry = str_count + 1;
        strncpy(strings[str_count].lexeme, start, copy_len); /* Copiamos sin comillas */
        strings[str_count].lexeme[copy_len] = '\0';

        return ++str_count;
    }


/* ------------------------------------------------------------
   FUNCIÓN: add_token
   ¿Qué hace?
     Agrega UN token al final de la lista de tokens.
     Esta función es llamada por cada regla de la sección 2
     cada vez que se reconoce un token en el código fuente.

   Parámetros:
     token_name: el nombre del token, ej: "PLUS", "DEF", "IDENTIFIER"
     entry:      0 si es keyword/operador/delimitador
                 número positivo si es literal (su fila en la tabla)
   ------------------------------------------------------------ */
    static void add_token(const char* token_name, int entry) {
        int id;

        /* Verificamos que la lista no esté llena */
        if (token_count >= MAX_ENTRIES * 8) {
            fprintf(stderr, "Error: lista de tokens llena.\n");
            exit(1);
        }

        /* Convertimos el nombre del token a su número ID */
        id = token_id_from_name(token_name);

        /* Si el nombre no se reconoció, hay un error de programación */
        if (id == 0) {
            fprintf(stderr, "Error interno: nombre de token desconocido: %s\n", token_name);
            exit(1);
        }

        /* Guardamos el token en la lista */
        token_list[token_count].token_id           = id;    /* El número del tipo de token */
        token_list[token_count].symbol_table_entry = entry; /* La fila en tabla (0 si no aplica) */
        token_count++; /* Avanzamos al siguiente espacio en la lista */
    }


/* ------------------------------------------------------------
   FUNCIÓN: print_results
   ¿Qué hace?
     Al terminar de analizar todo el código, esta función
     imprime el JSON completo con los tres resultados:
       1. token_ids    — el catálogo de IDs
       2. symbol_tables — las tres tablas de símbolos
       3. tokens        — la lista de tokens en orden

   El JSON se imprime a "stdout" que en main() redirigimos
   al archivo de salida.
   ------------------------------------------------------------ */
    static void print_results(void) {
        int i; /* Variable de control para los bucles */

        printf("{\n"); /* Abre el objeto JSON principal */

        /* --- Sección 1 del JSON: catálogo de IDs ---
           Lista completa que mapea número → nombre de token.
           Útil para que quien lea el JSON sepa qué significa cada número. */
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
        printf("  ],\n"); /* Cierra la sección token_ids */

        /* --- Sección 2 del JSON: tablas de símbolos --- */
        printf("  \"symbol_tables\": {\n");

        /* Tabla de identificadores — imprimimos cada fila como [número, "texto"] */
        printf("    \"identifiers\": [");
        for (i = 0; i < id_count; i++) {
            printf("[%d, ", identifiers[i].entry);        /* Número de entrada */
            print_json_string(identifiers[i].lexeme);     /* Texto del identificador */
            printf("]%s", i < id_count - 1 ? ", " : ""); /* Coma si no es el último */
        }
        printf("],\n");

        /* Tabla de números — cada fila como [número, "texto", "tipo"] */
        printf("    \"numbers\": [");
        for (i = 0; i < num_count; i++) {
            printf("[%d, ", numbers[i].entry);
            print_json_string(numbers[i].lexeme); /* El número como texto, ej: "3.14" */
            printf(", ");
            print_json_string(numbers[i].type);   /* El tipo: "int", "float" o "scientific" */
            printf("]%s", i < num_count - 1 ? ", " : "");
        }
        printf("],\n");

        /* Tabla de strings — cada fila como [número, "texto sin comillas"] */
        printf("    \"strings\": [");
        for (i = 0; i < str_count; i++) {
            printf("[%d, ", strings[i].entry);
            print_json_string(strings[i].lexeme);
            printf("]%s", i < str_count - 1 ? ", " : "");
        }
        printf("]\n"); /* Último elemento de symbol_tables, sin coma */

        printf("  },\n"); /* Cierra symbol_tables */

        /* --- Sección 3 del JSON: lista de tokens ---
           Cada token es [id] si es keyword/operador,
           o [id, entrada_en_tabla] si es un literal. */
        printf("  \"tokens\": [");
        for (i = 0; i < token_count; i++) {
            if (token_list[i].symbol_table_entry > 0)
                /* Literal: imprimimos el ID y la entrada en tabla */
                printf("[%d, %d]", token_list[i].token_id, token_list[i].symbol_table_entry);
            else
                /* Keyword/operador/delimitador: solo el ID */
                printf("[%d]", token_list[i].token_id);

            if (i < token_count - 1) printf(", "); /* Coma entre tokens */
        }
        printf("]\n"); /* Cierra el array de tokens */

        printf("}\n"); /* Cierra el objeto JSON principal */
    }

/* Fin de la sección %{ ... %}  — aquí termina el código C de la sección 1 */
%}


/* ============================================================
   OPCIONES DE FLEX
   ============================================================ */

/* noyywrap: por defecto Flex llama a yywrap() cuando termina el archivo
   para preguntar si hay más archivos que leer. Con "noyywrap" le decimos
   que no, solo hay un archivo. Sin esto el código no compilaría. */
/* yylineno: le pedimos a Flex que cuente las líneas automáticamente y
   guarde el número en la variable yylineno. La usamos en el mensaje de error. */
%option noyywrap yylineno


/* ============================================================
   DEFINICIONES DE PATRONES (macros de expresiones regulares)

   Aquí definimos "alias" para expresiones regulares complejas
   que vamos a usar en las reglas. Es como crear variables
   para no repetir el mismo patrón en múltiples lugares.

   Sintaxis: NOMBRE   patron
   Uso posterior en reglas: {NOMBRE}
   ============================================================ */

/* LETTER: cualquier letra (mayúscula o minúscula) o guión bajo _.
   Los identificadores en Python/Triton pueden empezar con letra o _. */
LETTER  [a-zA-Z_]

/* DIGIT: cualquier dígito del 0 al 9 */
DIGIT   [0-9]

/* ID: patrón para identificadores (nombres de variables/funciones).
   Regla: debe empezar con LETTER, seguido de cero o más LETTER o DIGIT.
   Ejemplos válidos: x, mi_var, BLOCK_SIZE, _privado, var123
   Ejemplos inválidos: 123var (empieza con número), mi-var (guión no permitido) */
ID      {LETTER}({LETTER}|{DIGIT})*

/* SCI: número en notación científica.
   Partes: dígitos, punto decimal OPCIONAL con más dígitos,
           letra e o E, signo + o - OPCIONAL, dígitos del exponente.
   Ejemplos: 1e5, 3.14e-2, 1E+10, 2.5e8 */
SCI     {DIGIT}+(\.[0-9]+)?[eE][+-]?{DIGIT}+

/* FLOAT: número con punto decimal (no científico).
   Tres formas válidas separadas por |:
     {DIGIT}+\.{DIGIT}+ → dígitos, punto, dígitos    (ej: 3.14)
     {DIGIT}+\.         → dígitos, punto solo         (ej: 3.)
     \.{DIGIT}+         → punto, dígitos              (ej: .5)  */
FLOAT   {DIGIT}+\.{DIGIT}+|{DIGIT}+\.|\.{DIGIT}+

/* INT: uno o más dígitos.
   Ejemplos: 0, 42, 1000 */
INT     {DIGIT}+

/* STR_DQ: string con comillas DOBLES.
   Partes: " al inicio,
           cero o más caracteres que sean: cualquier char excepto \, \n y "
           O una secuencia de escape (\ seguido de cualquier char),
           " al final.
   Esto permite strings como: "hola", "salto\n", "comilla\"" */
STR_DQ  \"([^\\\n\"]|\\.)*\"

/* STR_SQ: igual pero con comillas SIMPLES.
   Permite strings como: 'hola', 'inf', '-inf' */
STR_SQ  \'([^\\\n\']|\\.)*\'


/* ============================================================
   SECCIÓN 2 — REGLAS DE LEX
   ============================================================
   Esta es la sección más importante. Aquí definimos qué hacer
   cuando Flex encuentra cada patrón en el código fuente.

   Formato de cada regla:
     patron   { acción en C; }

   Flex lee el código fuente de izquierda a derecha y aplica
   la PRIMERA regla que coincida con el texto actual.
   Si dos reglas coinciden, gana la más LARGA (maximal munch).

   Por eso el ORDEN importa:
     - Keywords ANTES que identificadores (para que "def" sea
       DEF y no IDENTIFIER)
     - Operadores compuestos ANTES que simples ("==" antes que "=")
     - SCI y FLOAT ANTES que INT (para no consumir solo los dígitos)

   La variable especial "yytext" contiene el texto que acaba
   de ser reconocido por la regla actual.
   ============================================================ */

%%

 /* ------------------------------------------------------------
   ESPACIOS EN BLANCO — se ignoran, no generan tokens

   [ \t\r\n]+  significa: uno o más de: espacio, tab, retorno, salto de línea
   El punto y coma ; solo significa "no hacer nada".
   ------------------------------------------------------------ */
[ \t\r\n]+  ;   /* Espacio, tabulación, salto de línea → ignorar */

 /* ------------------------------------------------------------
   COMENTARIOS — se ignoran, no generan tokens

   #[^\n]*  significa: # seguido de cualquier carácter excepto salto de línea
   Todo lo que va después del # en una línea es comentario en Python/Triton.
   ------------------------------------------------------------ */
#[^\n]*     ;   /* Comentario: desde # hasta el fin de la línea → ignorar */


 /* ------------------------------------------------------------
   PALABRAS RESERVADAS

   IMPORTANTE: estas reglas van ANTES que la regla {ID}
   porque si {ID} fuera primero, la palabra "def" se reconocería
   como un IDENTIFIER en vez de como la keyword DEF.

   Flex aplica la regla más larga; si hay empate de longitud,
   aplica la que aparece primero en el archivo.

   add_token("DEF", 0) significa:
     - Agrega un token de tipo DEF a la lista
     - El 0 indica que no tiene entrada en tabla de símbolos
       (las keywords no se guardan en tablas, solo se cuentan)
   ------------------------------------------------------------ */
"def"       { add_token("DEF",      0); } /* Declaración de función           */
"for"       { add_token("FOR",      0); } /* Bucle for                        */
"if"        { add_token("IF",       0); } /* Condicional if                   */
"else"      { add_token("ELSE",     0); } /* Rama else del condicional        */
"elif"      { add_token("ELIF",     0); } /* Else-if encadenado               */
"while"     { add_token("WHILE",    0); } /* Bucle while                      */
"import"    { add_token("IMPORT",   0); } /* Importar módulo                  */
"as"        { add_token("AS",       0); } /* Alias en import                  */
"in"        { add_token("IN",       0); } /* Pertenencia en for               */
"True"      { add_token("TRUE",     0); } /* Booleano verdadero (con T mayúsc)*/
"False"     { add_token("FALSE",    0); } /* Booleano falso (con F mayúsc)    */
"None"      { add_token("NONE",     0); } /* Valor nulo (con N mayúsc)        */
"return"    { add_token("RETURN",   0); } /* Retornar valor                   */
"break"     { add_token("BREAK",    0); } /* Salir de bucle                   */
"continue"  { add_token("CONTINUE", 0); } /* Siguiente iteración              */
"pass"      { add_token("PASS",     0); } /* Bloque vacío                     */
"and"       { add_token("AND",      0); } /* Operador lógico Y                */
"or"        { add_token("OR",       0); } /* Operador lógico O                */
"not"       { add_token("NOT",      0); } /* Negación lógica                  */
"is"        { add_token("IS",       0); } /* Comparar identidad               */
"assert"    { add_token("ASSERT",   0); } /* Verificar condición              */
"from"      { add_token("FROM",     0); } /* from X import Y                  */
"try"       { add_token("TRY",      0); } /* Bloque try                       */
"except"    { add_token("EXCEPT",   0); } /* Capturar excepción               */
"global"    { add_token("GLOBAL",   0); } /* Declarar variable global         */
"raise"     { add_token("RAISE",    0); } /* Lanzar excepción                 */
"del"       { add_token("DEL",      0); } /* Eliminar variable                */


 /* ------------------------------------------------------------
   OPERADORES COMPUESTOS (dos caracteres)

   DEBEN ir ANTES que los operadores de un solo carácter.
   Razón (maximal munch): si el código dice "==", Flex intenta
   hacer coincidir el patrón más largo posible.
     - Si "=" está antes que "==", Flex reconoce "=" y luego
       otro "=" → dos tokens de asignación. INCORRECTO.
     - Si "==" está antes que "=", Flex reconoce "==" → un
       token de igualdad. CORRECTO.
   ------------------------------------------------------------ */
"**"        { add_token("POWER",         0); } /* Potencia: 2**3 = 8          */
"//"        { add_token("FLOOR_DIV",     0); } /* División entera: 7//2 = 3   */
"<<"        { add_token("LEFT_SHIFT",    0); } /* Desplazamiento izquierdo bits*/
">>"        { add_token("RIGHT_SHIFT",   0); } /* Desplazamiento derecho bits  */
"=="        { add_token("DOUBLE_EQUAL",  0); } /* Comparación de igualdad      */
"!="        { add_token("NOT_EQUAL",     0); } /* Diferente de                 */
"<="        { add_token("LESS_EQUAL",    0); } /* Menor o igual                */
">="        { add_token("GREATER_EQUAL", 0); } /* Mayor o igual                */


 /* ------------------------------------------------------------
   OPERADORES SIMPLES (un carácter)
   ------------------------------------------------------------ */
"+"         { add_token("PLUS",         0); } /* Suma                          */
"-"         { add_token("MINUS",        0); } /* Resta                         */
"*"         { add_token("MULTIPLY",     0); } /* Multiplicación                */
"/"         { add_token("DIVIDE",       0); } /* División                      */
"%"         { add_token("MODULO",       0); } /* Módulo / residuo              */
"="         { add_token("EQUAL",        0); } /* Asignación                    */
"<"         { add_token("LESS_THAN",    0); } /* Menor que                     */
">"         { add_token("GREATER_THAN", 0); } /* Mayor que                     */
"&"         { add_token("BIT_AND",      0); } /* AND a nivel de bits           */
"|"         { add_token("BIT_OR",       0); } /* OR a nivel de bits            */
"~"         { add_token("BIT_NOT",      0); } /* NOT a nivel de bits           */
"^"         { add_token("BIT_XOR",      0); } /* XOR a nivel de bits           */


 /* ------------------------------------------------------------
   DELIMITADORES (símbolos de estructura, no operadores)
   ------------------------------------------------------------ */
"."         { add_token("DOT",           0); } /* Punto: tl.load               */
","         { add_token("COMMA",         0); } /* Coma: separa argumentos      */
":"         { add_token("COLON",         0); } /* Dos puntos: inicio de bloque */
"@"         { add_token("AT_SIGN",       0); } /* Arroba: decorador @triton.jit*/
"("         { add_token("OPEN_PAREN",    0); } /* Paréntesis de apertura       */
")"         { add_token("CLOSE_PAREN",   0); } /* Paréntesis de cierre         */
"["         { add_token("OPEN_BRACKET",  0); } /* Corchete de apertura         */
"]"         { add_token("CLOSE_BRACKET", 0); } /* Corchete de cierre           */
"{"         { add_token("OPEN_BRACE",    0); } /* Llave de apertura            */
"}"         { add_token("CLOSE_BRACE",   0); } /* Llave de cierre              */


 /* ------------------------------------------------------------
   LITERALES (tokens que SÍ se guardan en tabla de símbolos)

   ORDEN OBLIGATORIO:
     1. STR_DQ y STR_SQ primero — para que las comillas no sean
        confundidas con otros patrones.
     2. SCI antes que FLOAT antes que INT — porque un número
        científico como "1e5" también coincide con INT (el "1").
        Si INT fuera primero, consumiría solo "1" y el "e5" daría error.

   add_string(yytext) y add_number(yytext, tipo) devuelven el
   número de entrada en la tabla. Ese número se pasa a add_token
   como segundo argumento para que quede registrado en el token.
   ------------------------------------------------------------ */
{STR_DQ}    { add_token("STRING",     add_string(yytext));               } /* String comillas dobles  */
{STR_SQ}    { add_token("STRING",     add_string(yytext));               } /* String comillas simples */
{SCI}       { add_token("SCIENTIFIC", add_number(yytext, "scientific")); } /* Número científico       */
{FLOAT}     { add_token("FLOAT",      add_number(yytext, "float"));      } /* Número flotante         */
{INT}       { add_token("INT",        add_number(yytext, "int"));        } /* Número entero           */
{ID}        { add_token("IDENTIFIER", add_identifier(yytext));           } /* Identificador           */


 /* ------------------------------------------------------------
   REGLA DE ERROR — captura cualquier carácter no reconocido

   El punto "." en Flex coincide con CUALQUIER carácter que
   no haya sido reconocido por ninguna regla anterior.
   Esto incluye caracteres como $, ?, !, etc. que no son
   parte del lenguaje Triton.

   yytext: el carácter inválido (Flex lo pone aquí automáticamente)
   yylineno: el número de línea actual (disponible gracias a %option yylineno)
   fprintf(stderr,...): imprime el error en la consola (no en el JSON)
   ------------------------------------------------------------ */
.           { fprintf(stderr, "Error léxico - carácter no reconocido: '%s' en la línea %d\n",
                      yytext, yylineno); }

%%
/* Fin de la sección 2 — aquí terminan las reglas de Flex */


/* ============================================================
   SECCIÓN 3 — CÓDIGO C DEL USUARIO
   Aquí va el main() y cualquier función extra.
   Este código se copia al final del archivo C generado por Flex.
   ============================================================ */

/* ------------------------------------------------------------
   FUNCIÓN PRINCIPAL: main
   ¿Qué hace?
     1. Verifica que se pasaron los argumentos correctos
        al ejecutar el programa desde la terminal.
     2. Abre el archivo de entrada (código Triton).
     3. Abre el archivo de salida (donde escribirá el JSON).
     4. Llama a yylex() — la función que Flex generó
        automáticamente para hacer el análisis léxico.
     5. Llama a print_results() para escribir el JSON.

   Uso desde terminal:
     ./lexer ejemplo.triton resultado.json
     ./lexer ejemplo.triton              ← usa "lexer_output.json" por defecto
   ------------------------------------------------------------ */
int main(int argc, char* argv[]) {
    /* argc: número de argumentos recibidos (incluyendo el nombre del programa)
       argv: arreglo de strings con cada argumento
       Ej: ./lexer ejemplo.triton salida.json
         argv[0] = "./lexer"
         argv[1] = "ejemplo.triton"
         argv[2] = "salida.json"
         argc    = 3 */

    const char* input_path;  /* Ruta del archivo de entrada  */
    const char* output_path; /* Ruta del archivo de salida   */

    /* Verificamos que el usuario pasó 1 o 2 argumentos (además del nombre del programa).
       Si pasó menos de 2 (falta el archivo de entrada) o más de 3, mostramos cómo usarlo. */
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Uso: %s <archivo_entrada.triton> [archivo_salida.json]\n", argv[0]);
        return 1; /* Retornar 1 indica que el programa terminó con ERROR */
    }

    /* Guardamos las rutas de los archivos */
    input_path  = argv[1]; /* El primer argumento es siempre el archivo de entrada */

    /* Si se dio un segundo argumento úsalo; si no, usa el nombre por defecto */
    output_path = (argc == 3) ? argv[2] : "lexer_output.json";

    /* freopen redirige stdin (entrada estándar) para que lea del archivo.
       Esto hace que yylex() lea del archivo en vez del teclado.
       Si falla (el archivo no existe), muestra error y termina. */
    if (freopen(input_path, "r", stdin) == NULL) {
        fprintf(stderr, "Error: no se pudo abrir el archivo de entrada: %s\n", input_path);
        return 1;
    }

    /* freopen redirige stdout (salida estándar) para que escriba al archivo.
       Esto hace que todos los printf() de print_results() vayan al JSON.
       Los fprintf(stderr,...) siguen yendo a la consola porque usan stderr. */
    if (freopen(output_path, "w", stdout) == NULL) {
        fprintf(stderr, "Error: no se pudo abrir el archivo de salida: %s\n", output_path);
        return 1;
    }

    /* yylex() es la función principal generada por Flex.
       Lee el archivo de entrada carácter por carácter,
       aplica las reglas de la sección 2 y llama add_token()
       cada vez que reconoce un token. Cuando termina el archivo, retorna. */
    yylex();

    /* Una vez que yylex() terminó, ya tenemos todas las tablas y la lista llenas.
       print_results() las formatea como JSON y las imprime a stdout
       (que ya está redirigido al archivo de salida). */
    print_results();

    /* Este mensaje va a stderr (la consola), no al archivo JSON.
       Sirve para confirmarle al usuario que el análisis terminó bien. */
    fprintf(stderr, "Listo. Resultado guardado en: %s\n", output_path);

    return 0; /* Retornar 0 indica que el programa terminó SIN ERRORES */
}
