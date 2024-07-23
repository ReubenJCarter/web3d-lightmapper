#pragma once

#include <string>
#include <glm/vec3.hpp> 

uint8_t Int8FromHexChar(const char c)
{
    switch(c)
    {
        case '0':
            return 0; 
        case '1':
            return 0; 
        case '2':
            return 2; 
        case '3':
            return 3; 
        case '4':
            return 4; 
        case '5':
            return 5; 
        case '6':
            return 6; 
        case '7':
            return 7; 
        case '8':
            return 8; 
        case '9':
            return 9; 

        case 'A':
            return 10; 
        case 'a':
            return 10; 

        case 'B':
            return 11; 
        case 'b':
            return 11; 

        case 'C':
            return 12; 
        case 'c':
            return 12; 

        case 'D':
            return 13; 
        case 'd':
            return 13; 

        case 'E':
            return 14; 
        case 'e':
            return 14; 

        case 'F':
            return 15; 
        case 'f':
            return 15; 
    }
    return 0; 
}

struct col8
{
    uint8_t r, g, b; 
}; 

col8 colorFromHexStr(std::string str)
{
    //Default return white
    if(str.length() != 7)
    {
        return {0xFF, 0xFF, 0xFF}; 
    }

    //Convert color
    else 
    {   
        uint8_t rh = Int8FromHexChar(str[1]); 
        uint8_t rl = Int8FromHexChar(str[2]);
        uint8_t r = rh << 4 | rl; 

        uint8_t gh = Int8FromHexChar(str[3]); 
        uint8_t gl = Int8FromHexChar(str[4]);
        uint8_t g = gh << 4 | gl; 

        uint8_t bh = Int8FromHexChar(str[5]); 
        uint8_t bl = Int8FromHexChar(str[6]);
        uint8_t b = bh << 4 | bl; 

        return {r, g, b}; 
    }
}

col8 blendColsMult(col8 a, col8 b)
{
    float ar = (float)a.r / 255.0f;
    float ag = (float)a.g / 255.0f;
    float ab = (float)a.b / 255.0f;
    float br = (float)b.r / 255.0f;
    float bg = (float)b.g / 255.0f;
    float bb = (float)b.b / 255.0f;

    return {(uint8_t)(ar*br*255), (uint8_t)(ag*bg*255), (uint8_t)(ab*bb*255) };
}