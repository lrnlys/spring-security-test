package com.rinanz.configuration;


import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Data
@Configuration
@ConfigurationProperties("application")
public class ApplicationProperties {

    private String ldapUrl;
    private String ldapUsername;
    private String ldapPassword;
    private String ldapBase;

}
