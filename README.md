# 📘 Guía Completa del Analizador Léxico — Triton Compiler
### Todo lo que necesitas saber para entender el proyecto y pasar el examen oral

---

## 📋 Tabla de Contenidos

1. [¿Qué es un compilador y por qué lo hacemos?](#1-qué-es-un-compilador-y-por-qué-lo-hacemos)
2. [¿Qué es el Analizador Léxico?](#2-qué-es-el-analizador-léxico)
3. [¿Qué es Triton?](#3-qué-es-triton)
4. [¿Qué herramientas usamos?](#4-qué-herramientas-usamos)
5. [Estructura del proyecto](#5-estructura-del-proyecto)
6. [Conceptos teóricos clave](#6-conceptos-teóricos-clave)
7. [Los tokens del lenguaje Triton](#7-los-tokens-del-lenguaje-triton)
8. [Las expresiones regulares explicadas](#8-las-expresiones-regulares-explicadas)
9. [El archivo .l explicado sección por sección](#9-el-archivo-l-explicado-sección-por-sección)
10. [Las tablas de símbolos](#10-las-tablas-de-símbolos)
11. [El JSON de salida](#11-el-json-de-salida)
12. [Cómo compilar y ejecutar](#12-cómo-compilar-y-ejecutar)
13. [Ejemplo completo paso a paso](#13-ejemplo-completo-paso-a-paso)
14. [Preguntas frecuentes del examen oral](#14-preguntas-frecuentes-del-examen-oral)

---

## 1. ¿Qué es un compilador y por qué lo hacemos?

Un **compilador** es un programa que traduce código escrito por humanos (como Python o C) a algo que la computadora puede ejecutar directamente.

El proceso de compilación tiene varias **fases** que se ejecutan en orden:

```
Código fuente
     ↓
[1. ANÁLISIS LÉXICO]     ← Nuestro proyecto
     ↓
[2. Análisis Sintáctico]
     ↓
[3. Análisis Semántico]
     ↓
[4. Generación de Código]
     ↓
Código ejecutable
```

**Nuestro proyecto** es únicamente la primera fase: el **Analizador Léxico** (también llamado *scanner* o *lexer*).

El objetivo final del compilador completo es **validar código Triton generado por modelos de inteligencia artificial (LLMs)** — verificar que el código sea léxicamente, sintácticamente y semánticamente correcto.

---

## 2. ¿Qué es el Analizador Léxico?

El analizador léxico es el primer paso de la compilación. Su trabajo es leer el código fuente **carácter por carácter** y agrupar esos caracteres en unidades con significado llamadas **tokens**.

### Analogía simple:

Imagina que el código fuente es una oración en español:
```
"El perro come huesos."
```

El analizador léxico la separa en palabras:
```
"El" | "perro" | "come" | "huesos" | "."
```

Igual con código Triton:
```python
x = 10 + 5
```
Se convierte en tokens:
```
IDENTIFIER(x) | EQUAL | INT(10) | PLUS | INT(5)
```

### ¿Qué hace exactamente nuestro lexer?

1. **Lee** un archivo `.triton` (código fuente)
2. **Reconoce** cada token usando patrones (expresiones regulares)
3. **Clasifica** cada token con un ID numérico
4. **Guarda** los literales (variables, números, strings) en tablas de símbolos
5. **Genera** un archivo JSON con toda esa información

### ¿Qué NO hace el lexer?

- No verifica si la gramática es correcta (eso es el parser/análisis sintáctico)
- No sabe si `x + +` tiene sentido (dos operadores seguidos)
- No sabe si una variable fue declarada antes de usarse (eso es análisis semántico)

---

## 3. ¿Qué es Triton?

**Triton** es un lenguaje de programación creado por OpenAI para escribir código que corre directamente en GPUs (tarjetas gráficas). Es un DSL (*Domain Specific Language* — lenguaje de propósito específico) basado en Python.

Se usa principalmente para escribir **kernels de GPU** — operaciones matemáticas masivamente paralelas como las que usan los modelos de inteligencia artificial.

### Ejemplo de código Triton:

```python
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(output_ptr + row_idx * n_cols + col_offsets, softmax_output, mask=mask)
```

Como ves, **la sintaxis es prácticamente Python** — por eso nuestro analizador léxico reconoce las mismas palabras clave que Python.

---

## 4. ¿Qué herramientas usamos?

### Flex (Fast Lexical Analyzer)

**Flex** es la herramienta principal del proyecto. Es un programa que:
1. Lee un archivo `.l` (que nosotros escribimos con las reglas)
2. **Genera automáticamente** un programa en C (`lex.yy.c`) que hace el análisis léxico

Es decir: **nosotros definimos las reglas, Flex crea el código**.

```
lexicalAnalyzer.l  →  [flex]  →  lex.yy.c  →  [gcc]  →  lexer.exe
   (nuestras          (genera)   (código C     (compila)  (ejecutable)
    reglas)                       generado)
```

### GCC (GNU Compiler Collection)

Es el compilador de C que convierte el `lex.yy.c` generado en un ejecutable.

### Por qué usamos Flex en vez de escribir el lexer a mano

Flex convierte internamente las expresiones regulares en **Autómatas Finitos Deterministas (DFA)**, que son extremadamente eficientes para reconocer patrones. Escribir eso a mano sería complejo y propenso a errores.

---

## 5. Estructura del proyecto

```
TritonCompiler/LexicalAnalyzer/
│
├── lexicalAnalyzer.l      ← El archivo PRINCIPAL que nosotros escribimos
│                            Contiene las reglas del analizador léxico en Flex
│
├── lex.yy.c               ← Generado automáticamente por Flex
│                            NO se edita a mano
│
├── lexer.exe              ← El programa ejecutable final
│                            Generado por GCC a partir de lex.yy.c
│
├── example.triton         ← Archivo de ejemplo con código Triton
│                            Sirve para probar el lexer
│
└── output_triton.json     ← El resultado: JSON con tokens y tablas
                             Generado al ejecutar el lexer
```

---

## 6. Conceptos teóricos clave

### 6.1 Token

Un **token** es la unidad mínima con significado en un lenguaje de programación.

Cada token tiene:
- Un **tipo** (qué clase de token es): `IF`, `IDENTIFIER`, `INT`, `PLUS`...
- Un **lexema** (el texto original): `if`, `x`, `42`, `+`
- Un **ID numérico** (cómo lo representamos): `3`, `28`, `31`, `33`

Ejemplo:
```
Código:    x = 42 + y
Tokens:    [IDENTIFIER "x"] [EQUAL] [INT "42"] [PLUS] [IDENTIFIER "y"]
En JSON:   [28,1]           [40]    [31,1]     [33]   [28,2]
```

### 6.2 Expresión Regular (Regex)

Una **expresión regular** es un patrón que describe un conjunto de strings.

| Símbolo | Significado | Ejemplo |
|---------|-------------|---------|
| `[abc]` | Cualquiera de esos caracteres | `[abc]` → a, b, o c |
| `[a-z]` | Rango de caracteres | `[a-z]` → cualquier minúscula |
| `.` | Cualquier carácter | `.` → a, 1, %, etc. |
| `*` | Cero o más repeticiones | `a*` → "", a, aa, aaa... |
| `+` | Una o más repeticiones | `a+` → a, aa, aaa... |
| `?` | Cero o una vez (opcional) | `a?` → "" o a |
| `\|` | O (alternativa) | `a\|b` → a o b |
| `()` | Agrupación | `(ab)+` → ab, abab... |
| `^` dentro de `[]` | Negación | `[^abc]` → cualquier cosa menos a, b, c |
| `\` | Escape (literal) | `\.` → el punto literal |

### 6.3 Autómata Finito Determinista (DFA/AFD)

Un DFA es un modelo matemático que:
- Tiene **estados** (representados como círculos)
- Lee una entrada **carácter por carácter**
- **Transiciona** de un estado a otro según el carácter leído
- Tiene estados de **aceptación** (donde reconoce un token)

Ejemplo simple — DFA para reconocer la palabra "if":

```
         i          f
→ [q0] ──→ [q1] ──→ [q2]✓
```

- `q0`: estado inicial
- `q1`: leímos una 'i'
- `q2`: leímos 'f' después de 'i' → **aceptamos** el token IF

**Flex convierte nuestras expresiones regulares en DFAs automáticamente.**

### 6.4 Maximal Munch (Coincidencia más larga)

Cuando Flex ve el texto `==`, hay dos posibilidades:
- Reconocer `=` y luego `=` (dos tokens EQUAL)
- Reconocer `==` (un token DOUBLE_EQUAL)

Flex siempre elige **la coincidencia más larga**. Por eso `==` se convierte en un solo token `DOUBLE_EQUAL`.

**Consecuencia importante:** Los operadores compuestos (`==`, `**`, `//`, etc.) deben estar **antes** que los simples (`=`, `*`, `/`) en el archivo `.l`.

### 6.5 Tabla de Símbolos

Una **tabla de símbolos** es una estructura de datos que guarda información sobre los elementos del código que necesitan recordarse para fases posteriores del compilador.

En nuestro caso tenemos tres tablas:
- **Identifiers**: nombres de variables y funciones
- **Numbers**: todos los números encontrados (con su tipo)
- **Strings**: todos los textos entre comillas

Las palabras clave y operadores **no** van a tabla de símbolos porque el compilador ya sabe todo sobre ellos — solo necesita saber que aparecieron.

---

## 7. Los tokens del lenguaje Triton

### Palabras Reservadas (IDs 1-27)

Son palabras que el lenguaje ya tiene definidas. No se pueden usar como nombres de variables.

| ID | Token | Texto | ¿Para qué sirve? |
|----|-------|-------|-----------------|
| 1 | DEF | `def` | Declarar funciones |
| 2 | FOR | `for` | Bucle for |
| 3 | IF | `if` | Condicional |
| 4 | ELSE | `else` | Rama alternativa |
| 5 | ELIF | `elif` | Else-if |
| 6 | WHILE | `while` | Bucle while |
| 7 | IMPORT | `import` | Importar módulos |
| 8 | AS | `as` | Alias en import |
| 9 | IN | `in` | Pertenencia |
| 10 | TRUE | `True` | Booleano verdadero |
| 11 | FALSE | `False` | Booleano falso |
| 12 | NONE | `None` | Valor nulo |
| 13 | RETURN | `return` | Retornar valor |
| 14 | BREAK | `break` | Salir de bucle |
| 15 | CONTINUE | `continue` | Siguiente iteración |
| 16 | PASS | `pass` | Bloque vacío |
| 17 | AND | `and` | Lógico Y |
| 18 | OR | `or` | Lógico O |
| 19 | NOT | `not` | Negación |
| 20 | IS | `is` | Comparar identidad |
| 21 | ASSERT | `assert` | Verificar condición |
| 22 | FROM | `from` | from X import Y |
| 23 | TRY | `try` | Bloque try |
| 24 | EXCEPT | `except` | Capturar error |
| 25 | GLOBAL | `global` | Variable global |
| 26 | RAISE | `raise` | Lanzar error |
| 27 | DEL | `del` | Eliminar variable |

### Identificadores y Literales (IDs 28-32)

Son tokens que SÍ se guardan en la tabla de símbolos porque su valor importa.

| ID | Token | Descripción | Ejemplo |
|----|-------|-------------|---------|
| 28 | IDENTIFIER | Nombre de variable o función | `x`, `mi_var`, `BLOCK_SIZE` |
| 29 | SCIENTIFIC | Número notación científica | `1e5`, `3.14e-2` |
| 30 | FLOAT | Número con decimal | `3.14`, `0.5`, `.25` |
| 31 | INT | Número entero | `0`, `42`, `1000` |
| 32 | STRING | Texto entre comillas | `"hola"`, `'inf'` |

### Operadores Aritméticos (IDs 33-39)

| ID | Token | Símbolo | Ejemplo |
|----|-------|---------|---------|
| 33 | PLUS | `+` | `a + b` |
| 34 | MINUS | `-` | `a - b` |
| 35 | MULTIPLY | `*` | `a * b` |
| 36 | DIVIDE | `/` | `a / b` |
| 37 | POWER | `**` | `2 ** 3` = 8 |
| 38 | FLOOR_DIV | `//` | `7 // 2` = 3 |
| 39 | MODULO | `%` | `7 % 2` = 1 |

### Operadores de Comparación y Asignación (IDs 40-46)

| ID | Token | Símbolo | Ejemplo |
|----|-------|---------|---------|
| 40 | EQUAL | `=` | `x = 5` (asignación) |
| 41 | LESS_THAN | `<` | `x < 10` |
| 42 | GREATER_THAN | `>` | `x > 10` |
| 43 | NOT_EQUAL | `!=` | `x != 5` |
| 44 | LESS_EQUAL | `<=` | `x <= 10` |
| 45 | GREATER_EQUAL | `>=` | `x >= 10` |
| 46 | DOUBLE_EQUAL | `==` | `x == 5` (comparación) |

### Operadores de Bits (IDs 47-52)

Operan directamente sobre los bits de los números. Muy usados en kernels GPU.

| ID | Token | Símbolo | Ejemplo binario |
|----|-------|---------|----------------|
| 47 | BIT_AND | `&` | `1010 & 1100` = `1000` |
| 48 | BIT_OR | `\|` | `1010 \| 0101` = `1111` |
| 49 | BIT_NOT | `~` | `~1010` = `0101` |
| 50 | BIT_XOR | `^` | `1010 ^ 1100` = `0110` |
| 51 | LEFT_SHIFT | `<<` | `1 << 3` = 8 |
| 52 | RIGHT_SHIFT | `>>` | `8 >> 3` = 1 |

### Delimitadores (IDs 53-62)

Dan estructura al código, no calculan nada.

| ID | Token | Símbolo | Uso |
|----|-------|---------|-----|
| 53 | DOT | `.` | Acceso a atributo: `tl.load` |
| 54 | COMMA | `,` | Separar argumentos |
| 55 | COLON | `:` | Inicio de bloque: `def f():` |
| 56 | AT_SIGN | `@` | Decorador: `@triton.jit` |
| 57 | OPEN_PAREN | `(` | Abre paréntesis |
| 58 | CLOSE_PAREN | `)` | Cierra paréntesis |
| 59 | OPEN_BRACKET | `[` | Abre corchete |
| 60 | CLOSE_BRACKET | `]` | Cierra corchete |
| 61 | OPEN_BRACE | `{` | Abre llave |
| 62 | CLOSE_BRACE | `}` | Cierra llave |

---

## 8. Las expresiones regulares explicadas

Estas son las expresiones regulares más importantes del proyecto, explicadas desde cero.

### LETTER — Cualquier letra o guión bajo
```
[a-zA-Z_]
```
- `a-z` → cualquier letra minúscula
- `A-Z` → cualquier letra mayúscula
- `_`   → guión bajo
- Los corchetes `[]` significan "cualquiera de estos"

### DIGIT — Cualquier dígito
```
[0-9]
```
Cualquier número del 0 al 9.

### ID — Identificadores (nombres de variables)
```
{LETTER}({LETTER}|{DIGIT})*
```
- `{LETTER}` → debe **empezar** con letra o guión bajo
- `(...)` → grupo
- `{LETTER}|{DIGIT}` → letra O dígito
- `*` → cero o más veces

Acepta: `x`, `mi_var`, `BLOCK_SIZE`, `var123`
Rechaza: `123var` (empieza con número), `mi-var` (guión no permitido)

### INT — Número entero
```
{DIGIT}+
```
- `{DIGIT}` → un dígito
- `+` → uno o más

Acepta: `0`, `42`, `1000`

### FLOAT — Número decimal
```
{DIGIT}+\.{DIGIT}+  |  {DIGIT}+\.  |  \.{DIGIT}+
```
Tres formas válidas separadas por `|`:
1. `3.14` → dígitos, punto, dígitos
2. `3.`   → dígitos, punto solo
3. `.14`  → punto, dígitos

El `\.` usa `\` para que el punto sea **literal** (sin `\`, el punto en regex significa "cualquier carácter").

### SCI — Notación científica
```
{DIGIT}+(\.[0-9]+)?[eE][+-]?{DIGIT}+
```
Partes:
- `{DIGIT}+` → uno o más dígitos
- `(\.[0-9]+)?` → punto y más dígitos, **opcional** (el `?` lo hace opcional)
- `[eE]` → la letra e o E
- `[+-]?` → signo más o menos, **opcional**
- `{DIGIT}+` → el exponente

Acepta: `1e5`, `3.14e-2`, `2E+10`, `1e05`

### STR_DQ — String con comillas dobles
```
\"([^\\\n\"]|\\.)*\"
```
- `\"` → comilla doble literal al inicio
- `[^\\\n\"]` → cualquier carácter EXCEPTO `\`, `\n` y `"`
- `|` → O
- `\\.` → barra invertida seguida de CUALQUIER carácter (secuencias de escape como `\n`, `\t`)
- `*` → cero o más de lo anterior
- `\"` → comilla doble literal al final

Acepta: `"hola"`, `"mundo\n"`, `"comilla\"escapada"`

### STR_SQ — String con comillas simples
```
\'([^\\\n\']|\\.)*\'
```
Igual que STR_DQ pero con comillas simples `'`.
Acepta: `'inf'`, `'-inf'`, `'hola'`

### Comentarios
```
#[^\n]*
```
- `#` → el símbolo hash
- `[^\n]*` → cero o más caracteres que NO sean salto de línea

Acepta todo desde `#` hasta el final de la línea.

---

## 9. El archivo .l explicado sección por sección

### Estructura general

Un archivo Flex (`.l`) tiene exactamente **tres secciones** separadas por `%%`:

```
%{
  SECCIÓN 1: Código C
  (includes, structs, variables, funciones)
%}
  definiciones de patrones
%%
  SECCIÓN 2: Reglas
  (patrón → acción)
%%
  SECCIÓN 3: Código C adicional
  (main y otras funciones)
```

### Sección 1 — El código C

#### Los structs (moldes de datos)
```c
typedef struct {
    int  entry;
    char lexeme[MAX_LEXEME_LEN];
} IdentifierEntry;
```
Este struct define cómo se ve **una fila** de la tabla de identificadores:
- `entry`: el número de fila (1, 2, 3...)
- `lexeme`: el texto del identificador

#### Las tablas de símbolos
```c
static IdentifierEntry identifiers[MAX_ENTRIES];
```
Un arreglo de hasta 4096 filas del tipo `IdentifierEntry`. Es la tabla completa.

#### El enum de IDs
```c
enum TokenId {
    TOK_DEF   = 1,
    TOK_FOR   = 2,
    ...
}
```
Asigna un número fijo a cada tipo de token. Así cuando el programa dice `TOK_PLUS`, todos saben que es el número 33.

#### La función add_token
```c
static void add_token(const char* token_name, int entry) { ... }
```
Esta es la función que se llama desde **cada regla** de la sección 2. Cada vez que Flex reconoce un token, esta función lo agrega a la lista.

#### La función print_results
```c
static void print_results(void) { ... }
```
Al final, esta función recorre todas las tablas y la lista de tokens y los imprime en formato JSON.

### Sección 2 — Las reglas

El formato de cada regla es:
```
patrón    { acción en C; }
```

#### Regla de espacios en blanco
```flex
[ \t\r\n]+  ;
```
El punto y coma solo significa "no hacer nada". Los espacios se ignoran.

#### Reglas de keywords
```flex
"def"   { add_token("DEF", 0); }
```
Cuando Flex ve exactamente `def`, llama `add_token` con nombre "DEF" y entrada 0 (sin tabla de símbolos).

#### Reglas de literales
```flex
{INT}   { add_token("INT", add_number(yytext, "int")); }
```
`yytext` es una **variable especial de Flex** que contiene el texto que acaba de ser reconocido. Aquí primero guardamos el número en la tabla con `add_number` (que devuelve el número de fila), y ese número se pasa a `add_token`.

#### La regla de error
```flex
.   { fprintf(stderr, "Error léxico: '%s' en línea %d\n", yytext, yylineno); }
```
El punto `.` en Flex coincide con **cualquier carácter no reconocido**. Esta regla siempre va **al final** porque si estuviera antes capturaría todo.

### Sección 3 — El main

```c
int main(int argc, char* argv[]) {
    freopen(input_path,  "r", stdin);   // Redirige lectura al archivo .triton
    freopen(output_path, "w", stdout);  // Redirige escritura al archivo .json
    yylex();           // Flex analiza todo el archivo
    print_results();   // Escribe el JSON
    return 0;
}
```

`freopen` "engaña" al programa haciéndole creer que su teclado es el archivo `.triton` y su pantalla es el archivo `.json`.

`yylex()` es la función principal que **Flex generó automáticamente** — nosotros nunca la escribimos, pero la llamamos.

---

## 10. Las tablas de símbolos

### ¿Por qué las necesitamos?

El analizador léxico no solo necesita decir "aquí hay un IDENTIFIER" — también necesita recordar **cuál** identificador es, para que las fases siguientes del compilador (sintaxis, semántica) puedan usarlo.

### ¿Qué va en cada tabla?

| Tabla | ¿Qué guarda? | ¿Por qué? |
|-------|-------------|-----------|
| identifiers | Variables y funciones | El parser necesita saber sus nombres |
| numbers | Todos los números | El parser necesita sus valores |
| strings | Textos entre comillas | El parser necesita su contenido |

### ¿Por qué keywords y operadores NO tienen tabla?

Porque el compilador ya sabe todo sobre `def` o `+` — están definidos por el lenguaje mismo. No hay nada nuevo que recordar sobre ellos.

### ¿Cómo evitamos duplicados?

Antes de agregar un elemento, la función busca si ya existe:
```c
for (i = 0; i < id_count; i++)
    if (strcmp(identifiers[i].lexeme, lexeme) == 0)
        return identifiers[i].entry; // Ya existe, devolvemos su número
```
Si `x` aparece 10 veces en el código, solo se guarda **una vez** en la tabla.

### Ejemplo de tablas para este código:
```python
x = 42 + x
```

**Tabla de identificadores:**
| entry | lexeme |
|-------|--------|
| 1 | x |

**Tabla de números:**
| entry | lexeme | type |
|-------|--------|------|
| 1 | 42 | int |

**Lista de tokens:**
```
[28, 1]  → IDENTIFIER, entrada 1 en tabla (x)
[40]     → EQUAL (=)
[31, 1]  → INT, entrada 1 en tabla (42)
[33]     → PLUS (+)
[28, 1]  → IDENTIFIER, entrada 1 en tabla (x de nuevo, misma entrada)
```

---

## 11. El JSON de salida

El resultado del lexer es un archivo JSON con esta estructura:

```json
{
  "token_ids": [
    [1, "DEF"], [2, "FOR"], [3, "IF"], ...
  ],
  "symbol_tables": {
    "identifiers": [
      [1, "x"], [2, "mi_funcion"]
    ],
    "numbers": [
      [1, "42", "int"], [2, "3.14", "float"]
    ],
    "strings": [
      [1, "hola mundo"]
    ]
  },
  "tokens": [
    [28, 1], [40], [31, 1], [33], [28, 1]
  ]
}
```

### Cómo leer un token de la lista

Cada elemento de `tokens` puede ser:
- `[40]` → Solo el ID. Significa: token tipo 40 (EQUAL), sin tabla de símbolos
- `[28, 1]` → ID y entrada. Significa: token tipo 28 (IDENTIFIER), entrada 1 en la tabla de identificadores

Para saber qué es el ID 28, buscamos en `token_ids` → `[28, "IDENTIFIER"]`.
Para saber qué identificador es el de entrada 1, buscamos en `symbol_tables.identifiers` → `[1, "x"]`.

---

## 12. Cómo compilar y ejecutar

### Paso 1: Generar el código C con Flex

```bash
flex lexicalAnalyzer.l
```

Esto genera el archivo `lex.yy.c` automáticamente.

### Paso 2: Compilar con GCC

```bash
gcc lex.yy.c -o lexer.exe
```

Esto genera el ejecutable `lexer.exe`.

### Paso 3: Ejecutar el lexer

```bash
./lexer.exe example.triton output.json
```

- `example.triton` → el código Triton que queremos analizar
- `output.json` → donde se guardará el resultado

Si no se especifica el segundo argumento, el resultado va a `lexer_output.json`:
```bash
./lexer.exe example.triton
```

### Flujo completo en un diagrama

```
lexicalAnalyzer.l
       ↓
   flex ← "Genera el código C"
       ↓
   lex.yy.c
       ↓
   gcc ← "Compila el C"
       ↓
   lexer.exe
       ↓
   ./lexer.exe example.triton output.json
       ↓
   output.json ← "Resultado final"
```

---

## 13. Ejemplo completo paso a paso

### Código de entrada (example.triton)
```python
@triton.jit
def add(x, y):
    return x + y
```

### ¿Qué hace el lexer?

Lee carácter por carácter. Cuando junta suficientes para hacer un token:

```
@       → AT_SIGN [56]
triton  → IDENTIFIER [28, 1]  (entrada 1 en tabla de identificadores)
.       → DOT [53]
jit     → IDENTIFIER [28, 2]  (entrada 2)
def     → DEF [1]             (keyword, sin tabla)
add     → IDENTIFIER [28, 3]  (entrada 3)
(       → OPEN_PAREN [57]
x       → IDENTIFIER [28, 4]  (entrada 4)
,       → COMMA [54]
y       → IDENTIFIER [28, 5]  (entrada 5)
)       → CLOSE_PAREN [58]
:       → COLON [55]
return  → RETURN [13]         (keyword, sin tabla)
x       → IDENTIFIER [28, 4]  (ya existe, devuelve entrada 4)
+       → PLUS [33]
y       → IDENTIFIER [28, 5]  (ya existe, devuelve entrada 5)
```

### Tablas resultantes

**Tabla de identificadores:**
| entry | lexeme |
|-------|--------|
| 1 | triton |
| 2 | jit |
| 3 | add |
| 4 | x |
| 5 | y |

**JSON de salida (fragmento):**
```json
{
  "symbol_tables": {
    "identifiers": [
      [1, "triton"], [2, "jit"], [3, "add"], [4, "x"], [5, "y"]
    ],
    "numbers": [],
    "strings": []
  },
  "tokens": [
    [56], [28,1], [53], [28,2],
    [1], [28,3], [57], [28,4], [54], [28,5], [58], [55],
    [13], [28,4], [33], [28,5]
  ]
}
```

---

## 14. Preguntas frecuentes del examen oral

### ¿Qué es un token?
La unidad mínima con significado en un lenguaje. Es la "palabra" del código. Tiene un tipo (como IDENTIFIER o PLUS), un texto original (lexema) y un ID numérico.

### ¿Qué es un lexema?
El texto exacto que aparece en el código fuente y que corresponde a un token. Por ejemplo, el lexema `42` corresponde al token INT.

### ¿Por qué usamos Flex y no escribimos el lexer a mano?
Flex convierte automáticamente nuestras expresiones regulares en Autómatas Finitos Deterministas (DFAs) optimizados. Hacerlo a mano sería muy complejo y propenso a errores.

### ¿Qué es yytext?
Es una variable global que Flex define automáticamente. Siempre contiene el texto del último token reconocido.

### ¿Qué es yylex()?
Es la función principal que Flex genera automáticamente. Lee el archivo de entrada, aplica las reglas y llama a las acciones correspondientes.

### ¿Por qué los operadores compuestos van antes que los simples?
Por el principio de maximal munch: Flex siempre intenta la coincidencia más larga. Si `=` estuviera antes que `==`, Flex reconocería dos tokens `=` en lugar de un `==`. Al poner `==` primero, Flex lo reconoce correctamente como un solo token.

### ¿Por qué SCI y FLOAT van antes que INT en las reglas?
Porque `3.14` también empieza con dígitos, igual que `3`. Si INT fuera primero, Flex consumiría solo el `3` y dejaría `.14` sin reconocer. Al poner FLOAT antes, Flex reconoce `3.14` completo.

### ¿Por qué las keywords van antes que IDENTIFIER?
Porque `def` también coincide con el patrón `{ID}` (letra seguida de letras). Si IDENTIFIER fuera primero, `def` sería tratado como un nombre de variable, no como la keyword DEF.

### ¿Por qué los comentarios no generan tokens?
Los comentarios son anotaciones para humanos. El compilador no los necesita para procesar el código, así que los ignoramos en la fase léxica.

### ¿Qué pasa si el lexer encuentra un carácter inválido como `$`?
La regla de error (el `.` al final) lo captura e imprime un mensaje de error en stderr. El análisis **continúa** — no se detiene — para poder reportar todos los errores.

### ¿Qué diferencia hay entre `=` y `==`?
- `=` es asignación: `x = 5` (le damos el valor 5 a x)
- `==` es comparación: `x == 5` (¿x vale 5?)

Son tokens distintos porque hacen cosas completamente diferentes.

### ¿Cuál es la diferencia entre INT, FLOAT y SCIENTIFIC?
- `INT`: número entero sin punto ni exponente: `42`, `0`, `100`
- `FLOAT`: número con punto decimal: `3.14`, `0.5`, `.25`
- `SCIENTIFIC`: número con exponente: `1e5`, `3.14e-2`

Los separamos porque el compilador puede necesitar tratarlos diferente en fases posteriores.

### ¿Qué es una tabla de símbolos?
Una estructura de datos que guarda información sobre los elementos del código que tienen un valor variable (identificadores, números, strings). Se usa en fases posteriores del compilador para verificar tipos, declaraciones, etc.

### ¿Por qué los identificadores no se duplican en la tabla?
Porque antes de agregar uno, la función `add_identifier` busca si ya existe. Si `x` aparece 5 veces en el código, solo hay una fila en la tabla. El token en la lista siempre apunta a esa misma fila.

### ¿Qué son los operadores de bits y para qué sirven en Triton?
Operan directamente sobre los bits de los números. En kernels GPU se usan para crear máscaras de bits eficientes, que son necesarias para operaciones paralelas sobre bloques de memoria.

### ¿Qué es `freopen`?
Redirige stdin o stdout hacia un archivo. Con `freopen(archivo, "r", stdin)` hacemos que `yylex()` lea de un archivo en vez del teclado. Con `freopen(archivo, "w", stdout)` hacemos que todos los `printf()` escriban al archivo JSON.

### ¿Cuáles son las tres secciones de un archivo .l?
1. **Definiciones**: código C (includes, structs, funciones) entre `%{` y `%}`, más definiciones de patrones
2. **Reglas**: patrones y sus acciones, entre `%%` y `%%`
3. **Código de usuario**: funciones adicionales y `main()` después del segundo `%%`

### ¿Qué hace exactamente el comando `flex lexicalAnalyzer.l`?
Lee el archivo `.l`, interpreta las expresiones regulares, construye internamente los DFAs correspondientes y genera el archivo `lex.yy.c` con el código C del analizador léxico listo para compilar.

### ¿Cuál es la diferencia entre `stderr` y `stdout`?
- `stdout`: salida estándar. Normalmente la pantalla, pero nosotros la redirigimos al archivo JSON con `freopen`.
- `stderr`: salida de errores. Siempre va a la consola. Los mensajes de error del lexer van aquí para no contaminar el JSON.

### ¿Por qué el proyecto usa JSON como salida?
JSON es un formato de datos estándar, fácil de leer tanto por humanos como por programas. Las fases siguientes del compilador (parser, analizador semántico) pueden leer el JSON y continuar el procesamiento.

---

*Proyecto desarrollado para TC3002B — Desarrollo de Aplicaciones Avanzadas de Ciencias Computacionales, ITESM Campus Guadalajara.*
