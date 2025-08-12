#!/usr/bin/env python3
"""
Module 0x05: Integrity Validation
CRC32: PLACEHOLDER_CRC32_05

FOR_EACH_MODULE: COMPUTE_AND_EMBED_CRC32
MASTER_MANIFEST: RECORD_ALL_CRC
VALIDATE_ON_START: TRUE
EXPECT: NO_ERROR
"""

import hashlib
import logging
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import zlib

logger = logging.getLogger(__name__)

class IntegrityValidator:
    """Integrity validation system for all modules"""
    
    def __init__(self):
        self.manifest_file = "integrity_manifest.json"
        self.module_checksums = {}
        self.expected_checksums = {}
        self.validation_results = {}
        self.is_valid_flag = False
        
        # Module definitions
        self.modules = {
            "main": "backend/main.py",
            "data_feed": "backend/modules/data_feed.py",
            "divergence_detector": "backend/modules/divergence_detector.py",
            "signal_engine": "backend/modules/signal_engine.py",
            "integrity_validation": "backend/modules/integrity_validation.py",
            "failsafe_execution": "backend/modules/failsafe_execution.py"
        }
        
        # Load existing manifest
        self._load_manifest()
        
        logger.info("Integrity Validator initialized")
    
    def _load_manifest(self):
        """Load existing integrity manifest"""
        try:
            if os.path.exists(self.manifest_file):
                with open(self.manifest_file, 'r') as f:
                    manifest_data = json.load(f)
                    self.expected_checksums = manifest_data.get("checksums", {})
                    logger.info("Loaded existing integrity manifest")
            else:
                logger.info("No existing manifest found, will create new one")
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            self.expected_checksums = {}
    
    def _compute_file_checksum(self, file_path: str) -> str:
        """Compute CRC32 checksum for a file"""
        try:
            if not os.path.exists(file_path):
                return "FILE_NOT_FOUND"
            
            with open(file_path, 'rb') as f:
                content = f.read()
                # Use zlib for CRC32 (more reliable than hashlib for this purpose)
                crc32 = zlib.crc32(content) & 0xffffffff
                return f"{crc32:08x}"
                
        except Exception as e:
            logger.error(f"Failed to compute checksum for {file_path}: {e}")
            return "ERROR"
    
    def _compute_module_checksum(self, module_path: str) -> str:
        """Compute checksum for a module and its dependencies"""
        try:
            if not os.path.exists(module_path):
                return "MODULE_NOT_FOUND"
            
            # Get all Python files in the module directory
            checksums = []
            
            if os.path.isfile(module_path):
                # Single file module
                checksums.append(self._compute_file_checksum(module_path))
            else:
                # Directory module
                for root, dirs, files in os.walk(module_path):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            checksums.append(self._compute_file_checksum(file_path))
            
            # Combine all checksums
            combined_content = "".join(checksums)
            combined_crc32 = zlib.crc32(combined_content.encode()) & 0xffffffff
            return f"{combined_crc32:08x}"
            
        except Exception as e:
            logger.error(f"Failed to compute module checksum for {module_path}: {e}")
            return "ERROR"
    
    def validate_all_modules(self) -> bool:
        """Validate integrity of all modules"""
        logger.info("Starting integrity validation...")
        
        all_valid = True
        validation_results = {}
        
        for module_name, module_path in self.modules.items():
            try:
                # Compute current checksum
                current_checksum = self._compute_module_checksum(module_path)
                self.module_checksums[module_name] = current_checksum
                
                # Get expected checksum
                expected_checksum = self.expected_checksums.get(module_name, "NO_EXPECTED")
                
                # Validate
                is_valid = current_checksum == expected_checksum
                validation_results[module_name] = {
                    "current_checksum": current_checksum,
                    "expected_checksum": expected_checksum,
                    "is_valid": is_valid,
                    "status": "VALID" if is_valid else "INVALID"
                }
                
                if not is_valid:
                    all_valid = False
                    logger.warning(f"Module {module_name} integrity check failed")
                    logger.warning(f"  Expected: {expected_checksum}")
                    logger.warning(f"  Current:  {current_checksum}")
                else:
                    logger.info(f"Module {module_name} integrity check passed")
                
            except Exception as e:
                logger.error(f"Error validating module {module_name}: {e}")
                validation_results[module_name] = {
                    "current_checksum": "ERROR",
                    "expected_checksum": "ERROR",
                    "is_valid": False,
                    "status": "ERROR"
                }
                all_valid = False
        
        self.validation_results = validation_results
        self.is_valid_flag = all_valid
        
        # Log validation summary
        if all_valid:
            logger.info("All modules passed integrity validation")
        else:
            logger.error("Some modules failed integrity validation")
        
        return all_valid
    
    def update_checksums(self):
        """Update checksums for all modules (used after code changes)"""
        logger.info("Updating module checksums...")
        
        for module_name, module_path in self.modules.items():
            checksum = self._compute_module_checksum(module_path)
            self.module_checksums[module_name] = checksum
            logger.info(f"Updated {module_name}: {checksum}")
        
        # Save to manifest
        self._save_manifest()
        logger.info("Checksums updated and saved to manifest")
    
    def _save_manifest(self):
        """Save integrity manifest to file"""
        try:
            manifest_data = {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "checksums": self.module_checksums,
                "modules": self.modules
            }
            
            with open(self.manifest_file, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            
            logger.info("Integrity manifest saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
    
    def get_validation_report(self) -> Dict:
        """Get detailed validation report"""
        return {
            "overall_valid": self.is_valid_flag,
            "validation_timestamp": datetime.now().isoformat(),
            "modules": self.validation_results,
            "checksums": self.module_checksums,
            "expected_checksums": self.expected_checksums
        }
    
    def get_module_status(self, module_name: str) -> Optional[Dict]:
        """Get status of a specific module"""
        if module_name in self.validation_results:
            return self.validation_results[module_name]
        return None
    
    def is_valid(self) -> bool:
        """Check if all modules are valid"""
        return self.is_valid_flag
    
    def get_integrity_score(self) -> float:
        """Get integrity score as percentage"""
        if not self.validation_results:
            return 0.0
        
        valid_count = sum(1 for result in self.validation_results.values() if result.get("is_valid", False))
        total_count = len(self.validation_results)
        
        return (valid_count / total_count) * 100.0 if total_count > 0 else 0.0
    
    def validate_single_module(self, module_name: str) -> bool:
        """Validate a single module"""
        if module_name not in self.modules:
            logger.error(f"Unknown module: {module_name}")
            return False
        
        module_path = self.modules[module_name]
        current_checksum = self._compute_module_checksum(module_path)
        expected_checksum = self.expected_checksums.get(module_name, "NO_EXPECTED")
        
        is_valid = current_checksum == expected_checksum
        
        # Update validation results
        self.validation_results[module_name] = {
            "current_checksum": current_checksum,
            "expected_checksum": expected_checksum,
            "is_valid": is_valid,
            "status": "VALID" if is_valid else "INVALID"
        }
        
        # Update overall validity
        if not is_valid:
            self.is_valid_flag = False
        
        return is_valid
    
    def force_validation(self) -> bool:
        """Force re-validation of all modules"""
        logger.info("Forcing integrity validation...")
        return self.validate_all_modules()
    
    def get_manifest_info(self) -> Dict:
        """Get information about the integrity manifest"""
        return {
            "manifest_file": self.manifest_file,
            "manifest_exists": os.path.exists(self.manifest_file),
            "last_updated": self.expected_checksums.get("_last_updated", "UNKNOWN"),
            "total_modules": len(self.modules),
            "validated_modules": len(self.validation_results)
        }
    
    def repair_checksums(self):
        """Repair checksums by updating them to current values"""
        logger.warning("Repairing checksums - this will update expected values to current values")
        
        # Recompute all checksums
        for module_name, module_path in self.modules.items():
            checksum = self._compute_module_checksum(module_path)
            self.module_checksums[module_name] = checksum
        
        # Save updated manifest
        self._save_manifest()
        
        # Re-validate
        self.validate_all_modules()
        
        logger.info("Checksums repaired and re-validated")
    
    def export_manifest(self, export_path: str = None) -> str:
        """Export integrity manifest to specified path"""
        if not export_path:
            export_path = f"integrity_manifest_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "export_version": "1.0.0",
                "checksums": self.module_checksums,
                "modules": self.modules,
                "validation_results": self.validation_results
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Integrity manifest exported to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Failed to export manifest: {e}")
            return ""
    
    def import_manifest(self, import_path: str) -> bool:
        """Import integrity manifest from file"""
        try:
            if not os.path.exists(import_path):
                logger.error(f"Import file not found: {import_path}")
                return False
            
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            # Validate import data structure
            required_keys = ["checksums", "modules"]
            if not all(key in import_data for key in required_keys):
                logger.error("Invalid manifest format")
                return False
            
            # Update checksums
            self.expected_checksums = import_data["checksums"]
            self.module_checksums = import_data["checksums"].copy()
            
            # Save to current manifest
            self._save_manifest()
            
            # Re-validate
            self.validate_all_modules()
            
            logger.info(f"Manifest imported from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import manifest: {e}")
            return False