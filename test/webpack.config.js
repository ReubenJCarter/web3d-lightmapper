const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
    mode: 'development',
	devtool: 'source-map',

    devServer: {
        static: './build',
    },
    
    output: {
        filename: '[name].bundle.js',
        path: path.resolve(__dirname, 'build'),
    },

    entry: {
        index: './test.js',
    },
    
    plugins: [
        new HtmlWebpackPlugin({
            hash: true,
            title: 'Test Web3D Light Mapper',
            template: './test.html',
        })
    ],

    module: {
        rules: [
            {
                test: /\.m?js$/,
                exclude: /(node_modules|bower_components)/,
                use: {
                    loader: 'babel-loader',
                    options: {
                        presets: ['@babel/preset-env'],
                    }
                }
            }
        ]
    },
}