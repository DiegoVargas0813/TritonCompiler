# Triton Compiler - Analizador Léxico

Este proyecto es el primer paso en la creación de un compilador para el lenguaje **Triton**. Su función principal es el **Análisis Léxico**, que consiste en transformar el código fuente (texto) en una lista de "tokens" (unidades con significado) que la computadora puede entender fácilmente.

## ¿Qué es un Analizador Léxico?
Imagina que estás leyendo una oración: `x = 10 + 5`. 
El analizador léxico separa esta oración en piezas individuales:
1. `x` -> Identificador (nombre de variable)
2. `=` -> Operador de asignación
3. `10` -> Número entero
4. `+` -> Operador de suma
5. `5` -> Número entero

## Estructura del Proyecto
- **[lexicalAnalyzer.l](file:///Users/santosa/Documents/GitHub/TritonCompiler/LexicalAnalyzer/lexicalAnalyzer.l)**: Es el archivo principal escrito en lenguaje Flex. Contiene las reglas que definen cómo reconocer palabras clave, números, strings y símbolos.
- **[example.triton](file:///Users/santosa/Documents/GitHub/TritonCompiler/LexicalAnalyzer/example.triton)**: Un archivo de ejemplo con código fuente de Triton para probar el analizador.
- **[lexer.exe](file:///Users/santosa/Documents/GitHub/TritonCompiler/LexicalAnalyzer/lexer.exe)**: El programa ejecutable (generado a partir del código C).

## Requisitos
Para compilar este proyecto desde cero, necesitas tener instalados:
1. **Flex**: Una herramienta que genera analizadores léxicos.
2. **GCC**: El compilador de lenguaje C.

## Cómo Compilar y Ejecutar

Si no sabes nada de programación, solo sigue estos pasos en tu terminal dentro de la carpeta `LexicalAnalyzer`:

### 1. Generar el código del analizador
Este comando toma las reglas de `lexicalAnalyzer.l` y crea un archivo de código en C llamado `lex.yy.c`.
```bash
flex lexicalAnalyzer.l
```

### 2. Compilar el programa
Este comando convierte el código C en un programa ejecutable llamado `lexer.exe`.
```bash
gcc lex.yy.c -o lexer.exe
```

### 3. Ejecutar el analizador
Para procesar un archivo de código Triton, usa el siguiente formato:
```bash
./lexer.exe <archivo_entrada.triton> <archivo_salida.json>
```
**Ejemplo:**
```bash
./lexer.exe example.triton output_triton.json
```

## ¿Qué obtengo como resultado?
El programa genera un archivo **JSON** (un formato de datos fácil de leer). Este archivo contiene:
- **token_ids**: Un catálogo de qué número corresponde a qué tipo de palabra (ej. `3` es `IF`).
- **symbol_tables**: Listas organizadas de todos los nombres de variables (identifiers), números y textos (strings) encontrados.
- **tokens**: La secuencia exacta de cómo aparece el código transformado en números.

## Reglas Soportadas
- **Comentarios**: Líneas que empiezan con `#` (son ignoradas).
- **Palabras Clave**: `def`, `if`, `else`, `for`, `while`.
- **Operadores**: `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`.
- **Símbolos**: `:`, `,`, `.`, `@`, `( )`, `[ ]`.
- **Números**: Enteros y decimales (ej. `10`, `3.14`).
- **Strings**: Texto entre comillas (ej. `"hola"`).
