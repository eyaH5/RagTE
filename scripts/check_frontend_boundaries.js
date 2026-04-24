const fs = require('fs');
const path = require('path');

const featuresDir = path.join(__dirname, '../frontend/src/features');

function checkBoundaries() {
    let hasErrors = false;

    function walk(dir) {
        const files = fs.readdirSync(dir);
        for (const file of files) {
            const filepath = path.join(dir, file);
            const stat = fs.statSync(filepath);
            
            if (stat.isDirectory()) {
                walk(filepath);
            } else if (file.endsWith('.ts') || file.endsWith('.tsx')) {
                const content = fs.readFileSync(filepath, 'utf-8');
                const lines = content.split('\n');
                
                lines.forEach((line, index) => {
                    const importMatch = line.match(/import\s+.*from\s+['"]([^'"]+)['"]/);
                    if (importMatch) {
                        const importPath = importMatch[1];
                        
                        // Check if it's a cross-feature import.
                        // A cross-feature import starts with '../' (meaning sibling feature)
                        if (importPath.startsWith('../') && !importPath.startsWith('../../')) {
                            const parts = importPath.replace('../', '').split('/');
                            const featureName = parts[0];
                            
                            // If it tries to import a specific file inside another feature instead of the barrel (index)
                            if (parts.length > 1) {
                                console.error(`FAIL: ${filepath.replace(__dirname, '')}:${index + 1}`);
                                console.error(`  Cross-feature internal import detected: '${importPath}'`);
                                console.error(`  Features must only import from other features via their barrel file (e.g., '../${featureName}').`);
                                hasErrors = true;
                            }
                        }
                    }
                });
            }
        }
    }

    if (fs.existsSync(featuresDir)) {
        walk(featuresDir);
    } else {
        console.log("No features directory found, skipping.");
    }

    if (hasErrors) {
        process.exit(1);
    } else {
        console.log("PASS: Frontend feature boundaries check passed! No internal cross-feature imports.");
        process.exit(0);
    }
}

checkBoundaries();
