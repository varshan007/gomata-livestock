const fs = require('fs');
const path = require('path');
const dirsToScan = ['src', 'routes', 'services', 'middleware', 'utils'];

function processFile(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    let original = content;

    const requiresLogger = content.includes("utils/logger");
    
    content = content.replace(/console\.log\(/g, "logger.info(");
    content = content.replace(/console\.error\(/g, "logger.error(");
    content = content.replace(/console\.warn\(/g, "logger.warn(");

    if (content !== original && !requiresLogger && !filePath.includes('logger.js')) {
        let relPath = path.relative(path.dirname(filePath), path.join(__dirname, 'utils/logger')).replace(/\\/g, '/');
        if (!relPath.startsWith('.')) relPath = './' + relPath;
        content = `const logger = require('${relPath}');\n` + content;
        fs.writeFileSync(filePath, content);
    } else if (content !== original) {
        fs.writeFileSync(filePath, content);
    }
}

function walkDir(dir) {
    if (!fs.existsSync(dir)) return;
    fs.readdirSync(dir).forEach(file => {
        let fullPath = path.join(dir, file);
        if (fs.lstatSync(fullPath).isDirectory()) {
            walkDir(fullPath);
        } else if (fullPath.endsWith('.js')) {
            processFile(fullPath);
        }
    });
}

dirsToScan.forEach(walkDir);
processFile('server.js');
console.log("Done");
